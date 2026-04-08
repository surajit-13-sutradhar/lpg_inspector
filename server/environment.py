"""
server/environment.py
LPG Inspector Environment — core logic.
Implements reset(), step(), and state property.
No networking here — pure business logic only.
"""
# -*- coding: utf-8 -*-

import uuid
from typing import Optional

from openenv.core.env_server import Environment

from models import (
    LPGInspectorAction,
    LPGInspectorObservation,
    LPGInspectorState,
    BatchContext,
)
from server.data_generator import (
    generate_cylinder,
    generate_easy_cylinder,
    generate_batch,
    generate_incident_scenario,
    WEIGHT_MIN_OK,
    WEIGHT_MAX_OK,
    VALVE_PRESSURE_MIN_BAR,
    VALVE_PRESSURE_MAX_BAR,
)
from server.graders import (
    SingleCylinderGrader,
    BatchInspectionGrader,
    IncidentRootCauseGrader,
)


# ─── Task constants ────────────────────────────────────────────────────────────

TASK_SINGLE   = "single_cylinder_triage"
TASK_BATCH    = "batch_inspection"
TASK_INCIDENT = "incident_root_cause"

VALID_TASKS = {TASK_SINGLE, TASK_BATCH, TASK_INCIDENT}

TASK_MAX_STEPS = {
    TASK_SINGLE:   5,
    TASK_BATCH:    10,
    TASK_INCIDENT: 15,
}

TASK_DIFFICULTY = {
    TASK_SINGLE:   "easy",
    TASK_BATCH:    "medium",
    TASK_INCIDENT: "hard",
}

VALID_DECISIONS = {"PASS", "FAIL", "RETEST", "QUARANTINE"}
VALID_PRIORITIES = {"NORMAL", "URGENT", "HOLD"}


# ─── Environment ──────────────────────────────────────────────────────────────

class LPGInspectorEnvironment(Environment):
    """
    LPG Cylinder Quality Control Inspector Environment.

    The agent acts as an AI quality control decision system sitting
    on top of an LPG cylinder inspection pipeline. Sensor readings
    and vision model outputs are pre-processed and presented as
    structured reports. The agent makes triage and dispatch decisions.

    Three tasks:
        single_cylinder_triage — easy
        batch_inspection       — medium
        incident_root_cause    — hard
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ─── Init ─────────────────────────────────────────────────────────────────

    def __init__(self):
        self._state        = LPGInspectorState()
        self._task_name    = TASK_SINGLE
        self._seed         = None

        # Task-specific data
        self._cylinder     = None      # single task
        self._batch        = []        # batch task
        self._scenario     = None      # incident task

        # Episode tracking
        self._decisions    = []        # all decisions this episode
        self._current_step = 0
        self._done         = False
        self._total_reward = 0.0
        self._safety_misses= 0

        # Graders
        self._grader1 = SingleCylinderGrader()
        self._grader2 = BatchInspectionGrader()
        self._grader3 = IncidentRootCauseGrader()

    # ─── reset() ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> LPGInspectorObservation:
        """
        Start a fresh episode.

        Args:
            seed       : Random seed for reproducibility
            episode_id : Optional episode identifier
            task_name  : Which task to run. One of:
                         'single_cylinder_triage' (default)
                         'batch_inspection'
                         'incident_root_cause'
        """
        # Resolve task
        self._task_name = task_name if task_name in VALID_TASKS else TASK_SINGLE
        self._seed      = seed

        # Reset all tracking
        self._decisions     = []
        self._current_step  = 0
        self._done          = False
        self._total_reward  = 0.0
        self._safety_misses = 0

        # Reset state
        max_steps = TASK_MAX_STEPS[self._task_name]
        self._state = LPGInspectorState(
            episode_id       = episode_id or str(uuid.uuid4()),
            step_count       = 0,
            task_name        = self._task_name,
            difficulty       = TASK_DIFFICULTY[self._task_name],
            max_steps        = max_steps,
            current_score    = 0.0,
            cylinders_total  = self._get_cylinders_total(),
            cylinders_done   = 0,
            batch_id         = "",
            safety_misses    = 0,
        )

        # Generate task data and return first observation
        if self._task_name == TASK_SINGLE:
            return self._reset_single()
        elif self._task_name == TASK_BATCH:
            return self._reset_batch()
        else:
            return self._reset_incident()

    def _reset_single(self) -> LPGInspectorObservation:
        """Reset for single cylinder triage task."""
        self._cylinder = generate_easy_cylinder(seed=self._seed)
        self._state.batch_id = self._cylinder["batch_id"]
        self._state.cylinders_total = 1

        return self._make_observation(
            cylinder      = self._cylinder,
            feedback      = (
                f"New episode started. Task: {TASK_SINGLE}.\n"
                f"Inspect the cylinder and submit your decision.\n"
                f"Nominal weight: {WEIGHT_MIN_OK} to {WEIGHT_MAX_OK} kg | "
                f"Safe pressure: {VALVE_PRESSURE_MIN_BAR} to {VALVE_PRESSURE_MAX_BAR} bar."
            ),
            progress      = 0.0,
        )

    def _reset_batch(self) -> LPGInspectorObservation:
        """Reset for batch inspection task."""
        self._batch = generate_batch(
            size       = 10,
            difficulty = "medium",
            seed       = self._seed,
        )
        self._state.batch_id = self._batch[0]["batch_id"] if self._batch else ""
        self._state.cylinders_total = len(self._batch)

        first_cylinder = self._batch[0]
        batch_ctx = BatchContext(
            batch_size          = len(self._batch),
            cylinders_processed = 0,
            cylinders_passed    = 0,
            cylinders_failed    = 0,
            cylinders_quarantine= 0,
            cylinders_retest    = 0,
            batch_pass_rate     = 0.0,
            batch_alerts        = [],
        )

        return self._make_observation(
            cylinder    = first_cylinder,
            feedback    = (
                f"New episode started. Task: {TASK_BATCH}.\n"
                f"Inspect all {len(self._batch)} cylinders in this batch.\n"
                f"Nominal weight: {WEIGHT_MIN_OK} to {WEIGHT_MAX_OK} kg | "
                f"Safe pressure: {VALVE_PRESSURE_MIN_BAR} to {VALVE_PRESSURE_MAX_BAR} bar."
            ),
            progress    = 0.0,
            batch_ctx   = batch_ctx,
        )

    def _reset_incident(self) -> LPGInspectorObservation:
        """Reset for incident root cause task."""
        self._scenario = generate_incident_scenario(seed=self._seed)
        self._state.cylinders_total = len(self._scenario["batch_ids"])

        # First step: show the incident report only
        # Agent must investigate across subsequent steps
        return LPGInspectorObservation(
            done               = False,
            reward             = None,
            cylinder_id        = "",
            batch_id           = "",
            weight_kg          = 0.0,
            valve_pressure_bar = 0.0,
            qr_status          = "",
            body_condition     = "",
            fill_date          = "",
            previous_failures  = 0,
            destination_zone   = self._scenario["affected_zone"],
            inspector_note     = "",
            incident_report    = self._scenario["incident_report"],
            available_batch_ids= self._scenario["batch_ids"],
            feedback_message   = (
                f"INCIDENT INVESTIGATION STARTED.\n"
                f"Review the incident report below.\n"
                f"Investigate the batches and identify the faulty one.\n"
                f"Submit your final answer as:\n"
                f"  decision     = the faulty batch ID\n"
                f"  defect_flags = list of batch IDs to recall\n"
                f"  reason       = root cause explanation\n"
                f"  priority     = URGENT"
            ),
            progress_score     = 0.0,
            task_name          = self._task_name,
            step_number        = 0,
            total_steps        = TASK_MAX_STEPS[self._task_name],
        )

    # ─── step() ───────────────────────────────────────────────────────────────

    def step(
        self,
        action: LPGInspectorAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> LPGInspectorObservation:
        """
        Process one action from the agent.
        """
        # Guard: episode already done
        if self._done:
            return self._terminal_observation(
                feedback="Episode is already complete. Call reset() to start a new episode."
            )

        self._current_step        += 1
        self._state.step_count    += 1
        self._state.cylinders_done = min(
            self._current_step, self._state.cylinders_total
        )

        # Validate action
        action = self._sanitize_action(action)

        # Route to correct task handler
        if self._task_name == TASK_SINGLE:
            return self._step_single(action)
        elif self._task_name == TASK_BATCH:
            return self._step_batch(action)
        else:
            return self._step_incident(action)

    def _step_single(self, action: LPGInspectorAction) -> LPGInspectorObservation:
        """Process one step for single cylinder triage."""
        # Record decision
        self._decisions.append(action.model_dump())

        # Grade immediately
        score = self._grader1.grade(
            decisions = self._decisions,
            cylinder  = self._cylinder,
            max_steps = TASK_MAX_STEPS[TASK_SINGLE],
        )

        reward = score
        self._total_reward   = reward
        self._state.current_score = reward

        # Check safety miss
        gt = self._cylinder["_ground_truth"]
        if action.decision == "PASS" and gt in {"FAIL", "QUARANTINE"}:
            self._safety_misses       += 1
            self._state.safety_misses  = self._safety_misses

        # Single task always ends after first decision
        self._done = True
        self._state.cylinders_done = 1

        feedback = self._build_single_feedback(action, score)

        return self._make_observation(
            cylinder  = self._cylinder,
            feedback  = feedback,
            progress  = score,
            reward    = reward,
            done      = True,
        )

    def _step_batch(self, action: LPGInspectorAction) -> LPGInspectorObservation:
        """Process one step for batch inspection."""
        idx = self._current_step - 1   # 0-indexed

        if idx >= len(self._batch):
            self._done = True
            return self._terminal_observation(
                feedback="All cylinders in batch have been processed."
            )

        current_cylinder = self._batch[idx]
        self._decisions.append(action.model_dump())

        # Check safety miss
        gt = current_cylinder["_ground_truth"]
        if action.decision == "PASS" and gt in {"FAIL", "QUARANTINE"}:
            self._safety_misses       += 1
            self._state.safety_misses  = self._safety_misses

        # Is this the last cylinder?
        is_last = (self._current_step >= len(self._batch))

        # Running score on decisions so far
        score = self._grader2.grade(
            decisions = self._decisions,
            cylinders = self._batch[:len(self._decisions)],
            max_steps = TASK_MAX_STEPS[TASK_BATCH],
        )
        self._state.current_score = score

        # Reward per step = incremental progress
        reward = score / max(1, self._current_step) * len(self._decisions)
        reward = min(max(reward, 0.0), 1.0)

        if is_last:
            self._done     = True
            self._total_reward = score
            reward         = score

        # Build batch context
        passed     = sum(1 for d in self._decisions if d["decision"] == "PASS")
        failed     = sum(1 for d in self._decisions if d["decision"] == "FAIL")
        quarantine = sum(1 for d in self._decisions if d["decision"] == "QUARANTINE")
        retest     = sum(1 for d in self._decisions if d["decision"] == "RETEST")
        alerts     = []
        if self._safety_misses > 0:
            alerts.append(f"WARNING: {self._safety_misses} potential safety miss(es) detected.")

        batch_ctx = BatchContext(
            batch_size           = len(self._batch),
            cylinders_processed  = len(self._decisions),
            cylinders_passed     = passed,
            cylinders_failed     = failed,
            cylinders_quarantine = quarantine,
            cylinders_retest     = retest,
            batch_pass_rate      = passed / max(1, len(self._decisions)),
            batch_alerts         = alerts,
        )

        # Next cylinder to show (or last if done)
        next_idx = self._current_step if not is_last else idx
        next_cyl = self._batch[min(next_idx, len(self._batch) - 1)]

        feedback = self._build_batch_feedback(action, current_cylinder, score, is_last)

        return self._make_observation(
            cylinder  = next_cyl,
            feedback  = feedback,
            progress  = len(self._decisions) / len(self._batch),
            reward    = reward,
            done      = is_last,
            batch_ctx = batch_ctx,
        )

    def _step_incident(self, action: LPGInspectorAction) -> LPGInspectorObservation:
        """Process one step for incident root cause task."""
        self._decisions.append(action.model_dump())

        max_steps = TASK_MAX_STEPS[TASK_INCIDENT]

        # Agent can investigate for multiple steps
        # Episode ends when agent submits a batch ID as decision
        # OR when max steps are reached
        submitted_decision = action.decision
        is_batch_id = submitted_decision in self._scenario["batch_ids"]
        is_last = is_batch_id or (self._current_step >= max_steps)

        if is_last:
            self._done = True
            score = self._grader3.grade(
                decisions = self._decisions,
                scenario  = self._scenario,
                max_steps = max_steps,
            )
            self._total_reward        = score
            self._state.current_score = score
            reward = score

            feedback = self._build_incident_feedback(action, score)

            return LPGInspectorObservation(
                done               = True,
                reward             = reward,
                cylinder_id        = "",
                batch_id           = "",
                weight_kg          = 0.0,
                valve_pressure_bar = 0.0,
                qr_status          = "",
                body_condition     = "",
                fill_date          = "",
                previous_failures  = 0,
                destination_zone   = self._scenario["affected_zone"],
                inspector_note     = "",
                incident_report    = self._scenario["incident_report"],
                available_batch_ids= self._scenario["batch_ids"],
                feedback_message   = feedback,
                progress_score     = score,
                task_name          = self._task_name,
                step_number        = self._current_step,
                total_steps        = max_steps,
            )

        # Still investigating — return scenario info again with progress
        progress = self._current_step / max_steps
        return LPGInspectorObservation(
            done               = False,
            reward             = 0.0,
            cylinder_id        = "",
            batch_id           = "",
            weight_kg          = 0.0,
            valve_pressure_bar = 0.0,
            qr_status          = "",
            body_condition     = "",
            fill_date          = "",
            previous_failures  = 0,
            destination_zone   = self._scenario["affected_zone"],
            inspector_note     = "",
            incident_report    = self._scenario["incident_report"],
            available_batch_ids= self._scenario["batch_ids"],
            feedback_message   = (
                f"Step {self._current_step}/{max_steps}. "
                f"Continue investigating or submit your final answer.\n"
                f"To submit: set decision=<batch_id>, "
                f"defect_flags=[<batch_ids_to_recall>], reason=<explanation>."
            ),
            progress_score     = progress,
            task_name          = self._task_name,
            step_number        = self._current_step,
            total_steps        = max_steps,
        )

    # ─── state property ───────────────────────────────────────────────────────

    @property
    def state(self) -> LPGInspectorState:
        return self._state

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _sanitize_action(self, action: LPGInspectorAction) -> LPGInspectorAction:
        """Clamp action fields to valid values."""
        decision = action.decision.upper().strip()

        # Allow batch IDs through for incident task
        valid_batch_ids = set(self._scenario["batch_ids"]) if self._scenario else set()

        if decision not in VALID_DECISIONS and decision not in valid_batch_ids:
            decision = "RETEST"   # Default to cautious on invalid input

        priority = action.priority.upper().strip()
        if priority not in VALID_PRIORITIES:
            priority = "NORMAL"

        return LPGInspectorAction(
            decision     = decision,
            reason       = (action.reason or "")[:500],
            defect_flags = [f.upper().strip() for f in action.defect_flags],
            priority     = priority,
        )

    def _get_cylinders_total(self) -> int:
        if self._task_name == TASK_SINGLE:
            return 1
        elif self._task_name == TASK_BATCH:
            return 10
        else:
            return 5   # approximate for incident task

    def _make_observation(
        self,
        cylinder:   dict,
        feedback:   str,
        progress:   float,
        reward:     Optional[float] = None,
        done:       bool = False,
        batch_ctx:  Optional[BatchContext] = None,
    ) -> LPGInspectorObservation:
        """Build a typed observation from a cylinder dict."""
        return LPGInspectorObservation(
            done               = done,
            reward             = reward,
            cylinder_id        = cylinder.get("cylinder_id", ""),
            batch_id           = cylinder.get("batch_id", ""),
            weight_kg          = cylinder.get("weight_kg", 0.0),
            valve_pressure_bar = cylinder.get("valve_pressure_bar", 0.0),
            qr_status          = cylinder.get("qr_status", "VALID"),
            body_condition     = cylinder.get("body_condition", "GOOD"),
            fill_date          = cylinder.get("fill_date", ""),
            previous_failures  = cylinder.get("previous_failures", 0),
            destination_zone   = cylinder.get("destination_zone", ""),
            inspector_note     = cylinder.get("inspector_note", ""),
            batch_context      = batch_ctx,
            feedback_message   = feedback,
            progress_score     = round(progress, 4),
            task_name          = self._task_name,
            step_number        = self._current_step,
            total_steps        = TASK_MAX_STEPS[self._task_name],
        )

    def _terminal_observation(self, feedback: str) -> LPGInspectorObservation:
        """Return a terminal observation when episode is already done."""
        return LPGInspectorObservation(
            done             = True,
            reward           = 0.0,
            cylinder_id      = "",
            batch_id         = "",
            weight_kg        = 0.0,
            valve_pressure_bar=0.0,
            qr_status        = "",
            body_condition   = "",
            fill_date        = "",
            previous_failures= 0,
            destination_zone = "",
            inspector_note   = "",
            feedback_message = feedback,
            progress_score   = self._state.current_score,
            task_name        = self._task_name,
            step_number      = self._current_step,
            total_steps      = TASK_MAX_STEPS[self._task_name],
        )

    def _build_single_feedback(
        self, action: LPGInspectorAction, score: float
    ) -> str:
        gt = self._cylinder["_ground_truth"]
        correct = action.decision == gt
        safety_miss = action.decision == "PASS" and gt in {"FAIL", "QUARANTINE"}

        if safety_miss:
            return (
                f"SAFETY MISS: You submitted PASS but ground truth was {gt}.\n"
                f"This is the most dangerous type of error.\n"
                f"Score: {score:.4f}"
            )
        elif correct:
            return (
                f"Correct decision: {action.decision}.\n"
                f"Score: {score:.4f}"
            )
        else:
            return (
                f" Incorrect. You submitted {action.decision}, "
                f"ground truth was {gt}.\n"
                f"Score: {score:.4f}"
            )

    def _build_batch_feedback(
        self,
        action: LPGInspectorAction,
        cylinder: dict,
        score: float,
        is_last: bool,
    ) -> str:
        gt = cylinder["_ground_truth"]
        correct = action.decision == gt
        step = self._current_step
        total = len(self._batch)

        status = "[CORRECT]" if correct else "[INCORRECT]"
        base = (
            f"{status} Cylinder {step}/{total}: "
            f"You said {action.decision}, ground truth was {gt}.\n"
            f"Running score: {score:.4f}"
        )
        if is_last:
            base += f"\nBatch complete. Final score: {score:.4f}"
        return base

    def _build_incident_feedback(
        self, action: LPGInspectorAction, score: float
    ) -> str:
        correct_batch = self._scenario["faulty_batch_id"]
        submitted     = action.decision
        correct       = submitted == correct_batch

        if correct:
            return (
                f"Correct batch identified: {submitted}.\n"
                f"Final score: {score:.4f}"
            )
        else:
            return (
                f" Incorrect batch. You submitted {submitted}, "
                f"faulty batch was {correct_batch}.\n"
                f"Final score: {score:.4f}"
            )