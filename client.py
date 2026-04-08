"""
client.py
HTTP/WebSocket client for the LPG Inspector environment.
This is what training code and inference scripts import.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import (
    LPGInspectorAction,
    LPGInspectorObservation,
    LPGInspectorState,
    BatchContext,
)


class LPGInspectorEnv(EnvClient[LPGInspectorAction, LPGInspectorObservation, LPGInspectorState]):
    """
    WebSocket/HTTP client for the LPG Inspector environment.

    Usage:
        # Sync (scripts, notebooks)
        with LPGInspectorEnv(base_url="https://...").sync() as env:
            result = env.reset(task_name="single_cylinder_triage")
            result = env.step(LPGInspectorAction(
                decision="QUARANTINE",
                reason="Valve pressure critically low.",
                defect_flags=["VALVE_PRESSURE_LOW"],
                priority="URGENT",
            ))
            print(result.observation.feedback_message)
            print(result.reward)
            print(result.done)

        # Async (training loops)
        async with LPGInspectorEnv(base_url="https://...") as env:
            result = await env.reset(task_name="batch_inspection")
            result = await env.step(action)
    """

    def _step_payload(self, action: LPGInspectorAction) -> dict:
        """Convert Action → dict for wire transmission."""
        return {
            "decision":     action.decision,
            "reason":       action.reason,
            "defect_flags": action.defect_flags,
            "priority":     action.priority,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Convert wire dict → StepResult with typed Observation."""
        obs_data = payload.get("observation", {})

        # Parse optional batch context
        batch_ctx_data = obs_data.get("batch_context")
        batch_ctx = None
        if batch_ctx_data:
            batch_ctx = BatchContext(
                batch_size           = batch_ctx_data.get("batch_size", 0),
                cylinders_processed  = batch_ctx_data.get("cylinders_processed", 0),
                cylinders_passed     = batch_ctx_data.get("cylinders_passed", 0),
                cylinders_failed     = batch_ctx_data.get("cylinders_failed", 0),
                cylinders_quarantine = batch_ctx_data.get("cylinders_quarantine", 0),
                cylinders_retest     = batch_ctx_data.get("cylinders_retest", 0),
                batch_pass_rate      = batch_ctx_data.get("batch_pass_rate", 0.0),
                batch_alerts         = batch_ctx_data.get("batch_alerts", []),
            )

        observation = LPGInspectorObservation(
            # Inherited fields
            done                = payload.get("done", False),
            reward              = payload.get("reward"),

            # Cylinder identification
            cylinder_id         = obs_data.get("cylinder_id", ""),
            batch_id            = obs_data.get("batch_id", ""),

            # Sensor readings
            weight_kg           = obs_data.get("weight_kg", 0.0),
            valve_pressure_bar  = obs_data.get("valve_pressure_bar", 0.0),
            qr_status           = obs_data.get("qr_status", ""),
            body_condition      = obs_data.get("body_condition", ""),

            # Context
            fill_date           = obs_data.get("fill_date", ""),
            previous_failures   = obs_data.get("previous_failures", 0),
            destination_zone    = obs_data.get("destination_zone", ""),
            inspector_note      = obs_data.get("inspector_note", ""),

            # Batch context
            batch_context       = batch_ctx,

            # Incident context
            incident_report     = obs_data.get("incident_report"),
            available_batch_ids = obs_data.get("available_batch_ids", []),

            # Episode feedback
            feedback_message    = obs_data.get("feedback_message", ""),
            progress_score      = obs_data.get("progress_score", 0.0),
            task_name           = obs_data.get("task_name", ""),
            step_number         = obs_data.get("step_number", 0),
            total_steps         = obs_data.get("total_steps", 0),
        )

        return StepResult(
            observation = observation,
            reward      = payload.get("reward"),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> LPGInspectorState:
        """Convert wire dict → typed State."""
        return LPGInspectorState(
            episode_id       = payload.get("episode_id"),
            step_count       = payload.get("step_count", 0),
            task_name        = payload.get("task_name", "single_cylinder_triage"),
            difficulty       = payload.get("difficulty", "easy"),
            max_steps        = payload.get("max_steps", 5),
            current_score    = payload.get("current_score", 0.0),
            cylinders_total  = payload.get("cylinders_total", 1),
            cylinders_done   = payload.get("cylinders_done", 0),
            batch_id         = payload.get("batch_id", ""),
            safety_misses    = payload.get("safety_misses", 0),
        )