"""
server/graders.py
Deterministic graders for all 3 LPG Inspector tasks.
Each grader receives episode history and returns a float in [0.0, 1.0].

Grading philosophy:
- Safety misses (FAIL/QUARANTINE labeled as PASS) are heavily penalized
- Partial credit for getting defect flags right even if decision is wrong
- Efficiency bonus for using fewer steps
- Always clamp final score to [0.0, 1.0]
"""

from typing import List, Dict, Any, Optional


# ─── Shared constants ─────────────────────────────────────────────────────────

VALID_DECISIONS = {"PASS", "FAIL", "RETEST", "QUARANTINE"}
VALID_FLAGS = {
    "WEIGHT_LOW", "WEIGHT_HIGH", "WEIGHT_BORDERLINE",
    "VALVE_PRESSURE_LOW", "VALVE_PRESSURE_HIGH",
    "QR_INVALID", "QR_MISSING", "QR_DUPLICATE",
    "BODY_DAMAGE", "SAFETY_HAZARD",
}
VALID_PRIORITIES = {"NORMAL", "URGENT", "HOLD"}

# Decisions that indicate safety risk if labeled PASS
SAFETY_CRITICAL_DECISIONS = {"FAIL", "QUARANTINE"}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _clamp(value: float) -> float:
    """Clamp score to [0.0, 1.0] and round to 4 decimal places."""
    return round(min(max(value, 0.0), 1.0), 4)


def _flag_score(submitted: List[str], expected: List[str]) -> float:
    """
    Compute F1-style score for defect flag detection.
    Returns 1.0 if perfect match, partial credit for partial match.
    Returns 1.0 if no flags expected AND none submitted (correct negative).
    """
    submitted_set = set(submitted)
    expected_set = set(expected)

    # Correct negative — no defects expected, none submitted
    if not expected_set and not submitted_set:
        return 1.0

    # No overlap at all
    if not expected_set or not submitted_set:
        return 0.0

    # Precision: of flags submitted, how many were correct?
    precision = len(submitted_set & expected_set) / len(submitted_set)

    # Recall: of expected flags, how many were found?
    recall = len(submitted_set & expected_set) / len(expected_set)

    # F1
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def _priority_score(submitted: str, expected: str) -> float:
    """
    Score priority assignment.
    Exact match = 1.0
    URGENT submitted but NORMAL expected = 0.5 (overcautious, not dangerous)
    NORMAL submitted but URGENT expected = 0.0 (dangerous miss)
    """
    if submitted == expected:
        return 1.0
    if expected == "URGENT" and submitted != "URGENT":
        return 0.0   # Missing urgency is dangerous
    if submitted == "URGENT" and expected == "NORMAL":
        return 0.5   # Overcautious but not dangerous
    return 0.3       # Other mismatches


def _is_safety_miss(submitted_decision: str, ground_truth: str) -> bool:
    """
    Returns True if agent said PASS but ground truth was FAIL or QUARANTINE.
    This is the most dangerous type of error.
    """
    return (
        submitted_decision == "PASS" and
        ground_truth in SAFETY_CRITICAL_DECISIONS
    )


def _decision_score(submitted: str, ground_truth: str) -> float:
    """
    Score the core decision.
    Exact match = 1.0
    Safety miss = 0.0 (handled separately with penalty)
    Adjacent decisions get partial credit.
    """
    if submitted == ground_truth:
        return 1.0

    # Safety miss — worst case
    if _is_safety_miss(submitted, ground_truth):
        return 0.0

    # Partial credit for adjacent decisions
    adjacency = {
        # (submitted, ground_truth): score
        ("RETEST", "FAIL"):       0.4,  # cautious but not right
        ("RETEST", "QUARANTINE"): 0.3,  # cautious but not right
        ("FAIL", "QUARANTINE"):   0.5,  # severity underestimated
        ("QUARANTINE", "FAIL"):   0.6,  # overcautious but safe
        ("PASS", "RETEST"):       0.2,  # missed ambiguity
        ("RETEST", "PASS"):       0.4,  # overcautious on clean cylinder
        ("FAIL", "PASS"):         0.0,  # false positive — penalized
    }
    return adjacency.get((submitted, ground_truth), 0.1)


# ─── Grader 1: Single Cylinder Triage (Easy Task) ────────────────────────────

class SingleCylinderGrader:
    """
    Grades performance on the single_cylinder_triage task.

    Episode structure:
        - 1 cylinder per episode
        - Agent gets up to 5 steps but should decide in 1
        - Score is based purely on decision quality

    Scoring weights:
        0.60 — core decision correctness
        0.20 — defect flag accuracy (F1)
        0.15 — priority assignment
        0.05 — efficiency (decided quickly)
        -0.40 — safety miss penalty (applied on top)
    """

    def grade(
        self,
        decisions: List[Dict[str, Any]],
        cylinder: Dict[str, Any],
        max_steps: int = 5,
    ) -> float:
        """
        Args:
            decisions : List of action dicts submitted by agent.
                        Each has: decision, reason, defect_flags, priority
            cylinder  : The cylinder dict from data_generator (has _ground_truth etc.)
            max_steps : Maximum steps allowed

        Returns:
            float in [0.0, 1.0]
        """
        if not decisions:
            return 0.0

        # Use the LAST decision (agent may have revised)
        final = decisions[-1]
        submitted_decision = final.get("decision", "PASS")
        submitted_flags    = final.get("defect_flags", [])
        submitted_priority = final.get("priority", "NORMAL")

        ground_truth     = cylinder["_ground_truth"]
        expected_flags   = cylinder["_expected_flags"]
        expected_priority= cylinder["_expected_priority"]

        # ── Component scores ──────────────────────────────────────────────────
        decision_sc = _decision_score(submitted_decision, ground_truth)
        flag_sc     = _flag_score(submitted_flags, expected_flags)
        priority_sc = _priority_score(submitted_priority, expected_priority)

        # Efficiency: fewer steps = small bonus
        steps_used  = len(decisions)
        efficiency_sc = max(0.0, (max_steps - steps_used) / max_steps)

        # ── Weighted sum ──────────────────────────────────────────────────────
        score = (
            0.60 * decision_sc +
            0.20 * flag_sc +
            0.15 * priority_sc +
            0.05 * efficiency_sc
        )

        # ── Safety miss penalty ───────────────────────────────────────────────
        if _is_safety_miss(submitted_decision, ground_truth):
            score -= 0.40

        return _clamp(score)


# ─── Grader 2: Batch Inspection (Medium Task) ─────────────────────────────────

class BatchInspectionGrader:
    """
    Grades performance on the batch_inspection task.

    Episode structure:
        - 10 cylinders per episode
        - Agent processes one cylinder per step
        - Score averages per-cylinder scores + dispatch order bonus

    Scoring weights per cylinder:
        0.50 — decision correctness
        0.25 — defect flag accuracy
        0.15 — priority assignment
        0.10 — dispatch order quality (urgent zones first)
        -0.30 — per safety miss

    Final score:
        0.80 — average per-cylinder score
        0.20 — batch-level dispatch order score
    """

    def grade(
        self,
        decisions: List[Dict[str, Any]],
        cylinders: List[Dict[str, Any]],
        max_steps: int = 10,
    ) -> float:
        """
        Args:
            decisions : List of action dicts, one per cylinder (in order)
            cylinders : List of cylinder dicts from data_generator
            max_steps : Maximum steps allowed

        Returns:
            float in [0.0, 1.0]
        """
        if not decisions or not cylinders:
            return 0.0

        # Pair decisions with cylinders (handle mismatched lengths)
        pairs = list(zip(decisions, cylinders))
        if not pairs:
            return 0.0

        per_cylinder_scores = []
        safety_miss_count = 0
        urgent_decisions = []   # Track order of URGENT decisions

        for decision_dict, cylinder in pairs:
            submitted_decision = decision_dict.get("decision", "PASS")
            submitted_flags    = decision_dict.get("defect_flags", [])
            submitted_priority = decision_dict.get("priority", "NORMAL")

            ground_truth      = cylinder["_ground_truth"]
            expected_flags    = cylinder["_expected_flags"]
            expected_priority = cylinder["_expected_priority"]

            # Per-cylinder component scores
            decision_sc = _decision_score(submitted_decision, ground_truth)
            flag_sc     = _flag_score(submitted_flags, expected_flags)
            priority_sc = _priority_score(submitted_priority, expected_priority)

            cyl_score = (
                0.50 * decision_sc +
                0.25 * flag_sc +
                0.15 * priority_sc
            )

            # Safety miss penalty
            if _is_safety_miss(submitted_decision, ground_truth):
                safety_miss_count += 1
                cyl_score -= 0.30

            per_cylinder_scores.append(_clamp(cyl_score))

            # Track urgency for dispatch ordering
            if submitted_priority == "URGENT":
                urgent_decisions.append(
                    (len(per_cylinder_scores), cylinder.get("destination_zone", ""))
                )

        avg_cylinder_score = sum(per_cylinder_scores) / len(per_cylinder_scores)

        # Dispatch order score:
        # Did agent correctly mark URGENT on all cylinders that needed it?
        needed_urgent_count = sum(
            1 for c in cylinders
            if c["_expected_priority"] == "URGENT"
        )
        correctly_marked_urgent = sum(
            1 for d, c in zip(decisions, cylinders)
            if c["_expected_priority"] == "URGENT"
            and d.get("priority") == "URGENT"
        )
        if needed_urgent_count == 0:
            dispatch_score = 1.0
        else:
            dispatch_score = correctly_marked_urgent / needed_urgent_count

        # Final weighted score
        score = (
            0.80 * avg_cylinder_score +
            0.20 * dispatch_score
        )

        return _clamp(score)


# ─── Grader 3: Incident Root Cause (Hard Task) ────────────────────────────────

class IncidentRootCauseGrader:
    """
    Grades performance on the incident_root_cause task.

    Episode structure:
        - Agent receives an incident report + multiple batch records
        - Must identify faulty batch, recall scope, root cause, corrective action
        - Up to 15 steps to investigate and submit final answer

    Scoring weights:
        0.40 — correct faulty batch identified
        0.30 — recall scope correct (not too wide, not too narrow)
        0.20 — root cause correctly identified
        0.10 — corrective action appropriate
        -0.20 — per incorrectly included batch in recall
    """

    # Root cause keywords we look for in agent's reason field
    ROOT_CAUSE_KEYWORDS = {
        "VALVE_LEAK":     ["valve", "pressure", "leak", "seal"],
        "RUST_CRITICAL":  ["rust", "corrosion", "corrode", "oxidat"],
        "MULTIPLE_ISSUES":["multiple", "several", "combined", "various"],
        "UNDERWEIGHT":    ["weight", "underfill", "fill", "mass"],
        "OVERWEIGHT":     ["overweight", "overfill", "excess"],
    }

    # Corrective action keywords
    CORRECTIVE_KEYWORDS = [
        "recall", "withdraw", "inspect", "retest",
        "quarantine", "isolate", "notify", "alert",
    ]

    def grade(
        self,
        decisions: List[Dict[str, Any]],
        scenario: Dict[str, Any],
        max_steps: int = 15,
    ) -> float:
        """
        Args:
            decisions : List of action dicts submitted by agent.
                        The LAST decision is treated as the final answer.
                        decision field = submitted faulty batch ID
                        defect_flags  = recall scope (list of batch IDs)
                        reason        = root cause explanation
                        priority      = urgency of response
            scenario  : The incident scenario dict from data_generator
            max_steps : Maximum steps allowed

        Returns:
            float in [0.0, 1.0]
        """
        if not decisions:
            return 0.0

        # Final answer is the last decision submitted
        final = decisions[-1]
        submitted_batch   = final.get("decision", "")
        submitted_scope   = final.get("defect_flags", [])  # batch IDs for recall
        submitted_reason  = final.get("reason", "").lower()
        submitted_priority= final.get("priority", "NORMAL")

        faulty_batch_id   = scenario["faulty_batch_id"]
        all_batch_ids     = scenario["batch_ids"]

        # ── 1. Correct faulty batch identified (0.40) ─────────────────────────
        batch_correct = 1.0 if submitted_batch == faulty_batch_id else 0.0

        # ── 2. Recall scope score (0.30) ──────────────────────────────────────
        # Correct scope = [faulty_batch_id] only
        # Too wide = includes clean batches (wastes resources)
        # Too narrow = misses faulty batch (dangerous)
        submitted_scope_set = set(submitted_scope)
        correct_scope_set   = {faulty_batch_id}

        if submitted_scope_set == correct_scope_set:
            scope_score = 1.0
        elif faulty_batch_id not in submitted_scope_set:
            scope_score = 0.0   # Missed the faulty batch entirely
        else:
            # Faulty batch included but too many others
            extra_batches = submitted_scope_set - correct_scope_set
            scope_score = max(0.0, 1.0 - 0.2 * len(extra_batches))

        # ── 3. Root cause identification (0.20) ───────────────────────────────
        # Check if reason contains relevant keywords
        # Identify what profiles are in the faulty batch
        faulty_batch = next(
            (b for b in scenario["batches"] if b["batch_id"] == faulty_batch_id),
            None
        )
        root_cause_score = 0.0
        if faulty_batch:
            faulty_profiles = {
                c["profile_name"]
                for c in faulty_batch["cylinders"]
                if c["profile_name"] != "PERFECT"
            }
            for profile in faulty_profiles:
                keywords = self.ROOT_CAUSE_KEYWORDS.get(profile, [])
                if any(kw in submitted_reason for kw in keywords):
                    root_cause_score = 1.0
                    break
            # Partial credit if they got something relevant
            if root_cause_score == 0.0:
                general_keywords = ["defect", "fault", "issue", "problem", "fail"]
                if any(kw in submitted_reason for kw in general_keywords):
                    root_cause_score = 0.3

        # ── 4. Corrective action (0.10) ───────────────────────────────────────
        corrective_score = 0.0
        if any(kw in submitted_reason for kw in self.CORRECTIVE_KEYWORDS):
            corrective_score = 1.0
        # Urgency bonus
        if submitted_priority == "URGENT":
            corrective_score = min(1.0, corrective_score + 0.2)

        # ── Weighted sum ──────────────────────────────────────────────────────
        score = (
            0.40 * batch_correct +
            0.30 * scope_score +
            0.20 * root_cause_score +
            0.10 * corrective_score
        )

        # ── Penalty: wrong batches included in recall ─────────────────────────
        # Cap penalty so recalling all batches still gives partial credit
        # if the faulty batch was correctly identified
        wrong_batches = submitted_scope_set - {faulty_batch_id}
        penalty = min(0.30, 0.10 * len(wrong_batches))
        score -= penalty

        return _clamp(score)