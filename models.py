"""
models.py
Pydantic type contracts for the LPG Inspector environment.
These are shared between client and server.
"""

from typing import List, Optional
from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel



# ─── Action ───────────────────────────────────────────────────────────────────

class LPGInspectorAction(Action):
    """
    What the agent submits for each cylinder inspection decision.

    Fields:
        decision     : Primary triage decision
                       'PASS'       — cylinder is good, approve for dispatch
                       'FAIL'       — cylinder fails inspection, reject
                       'RETEST'     — inconclusive, send for re-inspection
                       'QUARANTINE' — safety risk, isolate immediately
        reason       : Brief natural language justification (1-2 sentences)
        defect_flags : List of specific defects identified
                       Valid flags:
                         'WEIGHT_LOW'          — below minimum fill weight
                         'WEIGHT_HIGH'         — above maximum fill weight
                         'WEIGHT_BORDERLINE'   — within tolerance but flagged
                         'VALVE_PRESSURE_LOW'  — pressure below safe minimum
                         'VALVE_PRESSURE_HIGH' — pressure above safe maximum
                         'QR_INVALID'          — QR code unreadable
                         'QR_MISSING'          — no QR code present
                         'QR_DUPLICATE'        — QR ID already in system
                         'BODY_DAMAGE'         — dent or structural damage
                         'SAFETY_HAZARD'       — critical safety risk
        priority     : Dispatch priority if passed
                       'NORMAL' | 'URGENT' | 'HOLD'
    """
    decision:     str            = "PASS"
    reason:       str            = ""
    defect_flags: List[str]      = []
    priority:     str            = "NORMAL"


# ─── Observation ──────────────────────────────────────────────────────────────

class BatchContext(BaseModel):
    """
    Summary context about the current batch.
    Embedded inside LPGInspectorObservation for medium/hard tasks.
    """
    batch_size:           int        = 0
    cylinders_processed:  int        = 0
    cylinders_passed:     int        = 0
    cylinders_failed:     int        = 0
    cylinders_quarantine: int        = 0
    cylinders_retest:     int        = 0
    batch_pass_rate:      float      = 0.0
    batch_alerts:         List[str]  = []


class LPGInspectorObservation(Observation):
    """
    What the agent sees at each step — one cylinder's full inspection report.

    Inherited from Observation (DO NOT redefine):
        done:   bool
        reward: Optional[float]

    Cylinder identification:
        cylinder_id        — unique serial number e.g. 'CYL-2024-A-0042'
        batch_id           — batch this cylinder belongs to

    Sensor readings:
        weight_kg          — measured fill weight (nominal: 14.2 ± 0.15 kg)
        valve_pressure_bar — valve pressure reading (safe: 6.5–7.5 bar)
        qr_status          — 'VALID' | 'INVALID' | 'MISSING' | 'DUPLICATE'
        body_condition     — 'GOOD' | 'MINOR_DENT' | 'MAJOR_DENT' |
                             'RUST_MINOR' | 'RUST_CRITICAL'

    Context:
        fill_date          — date cylinder was filled (ISO format)
        previous_failures  — how many times this cylinder failed before
        destination_zone   — delivery zone assigned
        inspector_note     — automated system message

    Batch context (populated for medium/hard tasks):
        batch_context      — BatchContext object with batch-level stats

    Episode feedback:
        feedback_message   — environment's response to last action
        progress_score     — 0.0–1.0 progress indicator
        task_name          — which task is running
        step_number        — current step in episode
        total_steps        — total steps allowed for this task
    """

    # Cylinder identification
    cylinder_id:          str            = ""
    batch_id:             str            = ""

    # Sensor readings
    weight_kg:            float          = 0.0
    valve_pressure_bar:   float          = 0.0
    qr_status:            str            = "VALID"
    body_condition:       str            = "GOOD"

    # Context
    fill_date:            str            = ""
    previous_failures:    int            = 0
    destination_zone:     str            = ""
    inspector_note:       str            = ""

    # Batch context
    batch_context:        Optional[BatchContext] = None

    # Incident context (hard task only)
    incident_report:      Optional[str]  = None
    available_batch_ids:  List[str]      = []

    # Episode feedback
    feedback_message:     str            = ""
    progress_score:       float          = 0.0
    task_name:            str            = ""
    step_number:          int            = 0
    total_steps:          int            = 0


# ─── State ────────────────────────────────────────────────────────────────────

class LPGInspectorState(State):
    """
    Episode-level metadata.

    Inherited from State (DO NOT redefine):
        episode_id:  Optional[str]
        step_count:  int

    Fields:
        task_name        — 'single_cylinder_triage' | 'batch_inspection' |
                           'incident_root_cause'
        difficulty       — 'easy' | 'medium' | 'hard'
        max_steps        — maximum steps allowed this episode
        current_score    — running score (0.0–1.0)
        cylinders_total  — total cylinders to process
        cylinders_done   — cylinders processed so far
        batch_id         — active batch ID
        safety_misses    — count of FAIL-labeled-as-PASS decisions (critical)
    """
    task_name:        str    = "single_cylinder_triage"
    difficulty:       str    = "easy"
    max_steps:        int    = 5
    current_score:    float  = 0.0
    cylinders_total:  int    = 1
    cylinders_done:   int    = 0
    batch_id:         str    = ""
    safety_misses:    int    = 0