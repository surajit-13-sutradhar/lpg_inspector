"""
server/data_generator.py
Synthetic LPG cylinder inspection data generator.
All data is procedurally generated — no external datasets needed.
Seeded randomness ensures reproducibility.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional


# ─── Physical constants & thresholds ──────────────────────────────────────────

WEIGHT_NOMINAL_KG       = 14.2    # Expected fill weight
WEIGHT_TOLERANCE_KG     = 0.15    # ± acceptable range
WEIGHT_MIN_OK           = WEIGHT_NOMINAL_KG - WEIGHT_TOLERANCE_KG   # 14.05
WEIGHT_MAX_OK           = WEIGHT_NOMINAL_KG + WEIGHT_TOLERANCE_KG   # 14.35

VALVE_PRESSURE_MIN_BAR  = 6.5
VALVE_PRESSURE_MAX_BAR  = 7.5
VALVE_NOMINAL_BAR       = 7.0

ZONES = [
    "ZONE_A_HIGH_DEMAND",
    "ZONE_B_RESIDENTIAL",
    "ZONE_C_COMMERCIAL",
    "ZONE_D_RURAL",
    "ZONE_E_INDUSTRIAL",
]

BODY_CONDITIONS = [
    "GOOD",
    "MINOR_DENT",
    "MAJOR_DENT",
    "RUST_MINOR",
    "RUST_CRITICAL",
]

QR_STATUSES = [
    "VALID",
    "INVALID",
    "MISSING",
    "DUPLICATE",
]

# ─── Cylinder profile types ────────────────────────────────────────────────────
# Each profile has a name, ground truth decision, expected defect flags,
# and parameter overrides that make the reading realistic.

CYLINDER_PROFILES = {
    "PERFECT": {
        "ground_truth":  "PASS",
        "defect_flags":  [],
        "priority":      "NORMAL",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "UNDERWEIGHT": {
        "ground_truth":  "FAIL",
        "defect_flags":  ["WEIGHT_LOW"],
        "priority":      "URGENT",
        "weight_range":  (12.5, 14.04),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "OVERWEIGHT": {
        "ground_truth":  "FAIL",
        "defect_flags":  ["WEIGHT_HIGH"],
        "priority":      "URGENT",
        "weight_range":  (14.36, 15.5),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "VALVE_LEAK": {
        "ground_truth":  "QUARANTINE",
        "defect_flags":  ["VALVE_PRESSURE_LOW"],
        "priority":      "URGENT",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(4.0, 6.4),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 1,
    },
    "VALVE_OVERPRESSURE": {
        "ground_truth":  "QUARANTINE",
        "defect_flags":  ["VALVE_PRESSURE_HIGH"],
        "priority":      "URGENT",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(7.6, 9.5),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "BAD_QR": {
        "ground_truth":  "RETEST",
        "defect_flags":  ["QR_INVALID"],
        "priority":      "NORMAL",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "INVALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "MISSING_QR": {
        "ground_truth":  "RETEST",
        "defect_flags":  ["QR_MISSING"],
        "priority":      "NORMAL",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "MISSING",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "DUPLICATE_QR": {
        "ground_truth":  "QUARANTINE",
        "defect_flags":  ["QR_DUPLICATE"],
        "priority":      "URGENT",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "DUPLICATE",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "MAJOR_DENT": {
        "ground_truth":  "FAIL",
        "defect_flags":  ["BODY_DAMAGE"],
        "priority":      "URGENT",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"MAJOR_DENT",
        "prev_failures": 0,
    },
    "RUST_CRITICAL": {
        "ground_truth":  "QUARANTINE",
        "defect_flags":  ["SAFETY_HAZARD", "BODY_DAMAGE"],
        "priority":      "URGENT",
        "weight_range":  (14.10, 14.30),
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"RUST_CRITICAL",
        "prev_failures": 2,
    },
    "BORDERLINE_WEIGHT": {
        "ground_truth":  "RETEST",
        "defect_flags":  ["WEIGHT_BORDERLINE"],
        "priority":      "NORMAL",
        "weight_range":  (14.04, 14.06),   # Right on the edge
        "pressure_range":(6.8, 7.3),
        "qr_status":     "VALID",
        "body_condition":"GOOD",
        "prev_failures": 0,
    },
    "MULTIPLE_ISSUES": {
        "ground_truth":  "QUARANTINE",
        "defect_flags":  ["WEIGHT_LOW", "VALVE_PRESSURE_LOW", "BODY_DAMAGE"],
        "priority":      "URGENT",
        "weight_range":  (12.5, 14.04),
        "pressure_range":(4.0, 6.4),
        "qr_status":     "VALID",
        "body_condition":"MAJOR_DENT",
        "prev_failures": 3,
    },
}

# ─── Easy task profiles (clear-cut, no ambiguity) ─────────────────────────────
EASY_PROFILES = [
    "PERFECT",
    "UNDERWEIGHT",
    "OVERWEIGHT",
    "VALVE_LEAK",
    "BAD_QR",
    "MAJOR_DENT",
    "RUST_CRITICAL",
]

# ─── Medium task profiles (mix of easy + borderline) ─────────────────────────
MEDIUM_PROFILES = EASY_PROFILES + [
    "BORDERLINE_WEIGHT",
    "MISSING_QR",
    "DUPLICATE_QR",
    "VALVE_OVERPRESSURE",
]

# ─── Hard task profiles (all + multiple issues) ───────────────────────────────
HARD_PROFILES = MEDIUM_PROFILES + [
    "MULTIPLE_ISSUES",
]


# ─── Inspector notes — realistic automated system messages ───────────────────

INSPECTOR_NOTES = {
    "PERFECT":           "All readings nominal. Cleared by automated system.",
    "UNDERWEIGHT":       "Weight sensor alert: reading below minimum threshold.",
    "OVERWEIGHT":        "Weight sensor alert: reading exceeds maximum threshold.",
    "VALVE_LEAK":        "Pressure sensor alert: valve pressure drop detected.",
    "VALVE_OVERPRESSURE":"Pressure sensor alert: valve pressure exceeds safe limit.",
    "BAD_QR":            "QR scanner: code unreadable after 3 attempts.",
    "MISSING_QR":        "QR scanner: no QR code detected on cylinder.",
    "DUPLICATE_QR":      "QR scanner: ID already registered to another cylinder in system.",
    "MAJOR_DENT":        "Vision system: structural deformation detected on cylinder body.",
    "RUST_CRITICAL":     "Vision system: severe corrosion detected. Safety risk flagged.",
    "BORDERLINE_WEIGHT": "Weight sensor: reading within 10g of minimum threshold. Flagged for review.",
    "MULTIPLE_ISSUES":   "Multiple sensor alerts triggered. Manual inspection recommended.",
}


# ─── Core generator functions ─────────────────────────────────────────────────

def _rng(seed: Optional[int] = None) -> random.Random:
    """Return a seeded Random instance."""
    return random.Random(seed)


def generate_cylinder_id() -> str:
    """Generate a realistic cylinder serial number."""
    return f"CYL-{datetime.now().year}-{random.choice('ABCDEFGH')}-{random.randint(1000, 9999)}"


def generate_batch_id() -> str:
    """Generate a realistic batch ID."""
    today = datetime.now().strftime("%Y%m%d")
    shift = random.randint(1, 3)
    return f"BATCH-{today}-S{shift:02d}"


def generate_fill_date(days_ago_max: int = 5) -> str:
    """Generate a recent fill date."""
    days_ago = random.randint(0, days_ago_max)
    fill_dt = datetime.now() - timedelta(days=days_ago)
    return fill_dt.strftime("%Y-%m-%d")


def generate_cylinder(
    profile_name: Optional[str] = None,
    seed: Optional[int] = None,
    cylinder_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> dict:
    """
    Generate one cylinder inspection report.

    Args:
        profile_name: One of CYLINDER_PROFILES keys. If None, chosen randomly.
        seed:         Random seed for reproducibility.
        cylinder_id:  Override cylinder ID (for batch consistency).
        batch_id:     Override batch ID (for batch consistency).

    Returns:
        dict with all cylinder fields + ground_truth metadata.
    """
    rng = _rng(seed)

    if profile_name is None:
        profile_name = rng.choice(list(CYLINDER_PROFILES.keys()))

    profile = CYLINDER_PROFILES[profile_name]

    # Generate readings from profile ranges
    weight_kg = round(
        rng.uniform(*profile["weight_range"]), 3
    )
    valve_pressure_bar = round(
        rng.uniform(*profile["pressure_range"]), 2
    )

    return {
        # Identifiers
        "cylinder_id":          cylinder_id or generate_cylinder_id(),
        "batch_id":             batch_id or generate_batch_id(),
        "profile_name":         profile_name,

        # Sensor readings
        "weight_kg":            weight_kg,
        "valve_pressure_bar":   valve_pressure_bar,
        "qr_status":            profile["qr_status"],
        "body_condition":       profile["body_condition"],

        # Contextual info
        "fill_date":            generate_fill_date(),
        "previous_failures":    profile["prev_failures"],
        "destination_zone":     rng.choice(ZONES),
        "inspector_note":       INSPECTOR_NOTES[profile_name],

        # Ground truth (used by graders, NOT shown to agent directly)
        "_ground_truth":        profile["ground_truth"],
        "_expected_flags":      profile["defect_flags"],
        "_expected_priority":   profile["priority"],
    }


def generate_easy_cylinder(seed: Optional[int] = None) -> dict:
    """Generate a cylinder using only easy (clear-cut) profiles."""
    rng = _rng(seed)
    profile = rng.choice(EASY_PROFILES)
    return generate_cylinder(profile_name=profile, seed=seed)


def generate_batch(
    size: int = 10,
    difficulty: str = "medium",
    seed: Optional[int] = None,
    batch_id: Optional[str] = None,
) -> list[dict]:
    """
    Generate a batch of cylinders.

    Args:
        size:       Number of cylinders in batch.
        difficulty: 'easy' | 'medium' | 'hard'
        seed:       Random seed for reproducibility.
        batch_id:   Shared batch ID for all cylinders.

    Returns:
        List of cylinder dicts.
    """
    rng = _rng(seed)
    batch_id = batch_id or generate_batch_id()

    profiles_pool = {
        "easy":   EASY_PROFILES,
        "medium": MEDIUM_PROFILES,
        "hard":   HARD_PROFILES,
    }.get(difficulty, MEDIUM_PROFILES)

    cylinders = []
    for i in range(size):
        profile = rng.choice(profiles_pool)
        cyl_seed = rng.randint(0, 999999)
        cyl = generate_cylinder(
            profile_name=profile,
            seed=cyl_seed,
            cylinder_id=f"CYL-{datetime.now().year}-{chr(65 + (i % 8))}-{1000 + i}",
            batch_id=batch_id,
        )
        cylinders.append(cyl)

    return cylinders


def generate_incident_scenario(seed: Optional[int] = None) -> dict:
    """
    Generate a multi-batch incident scenario for the hard task.

    One batch is 'faulty' (contains VALVE_LEAK + RUST_CRITICAL cylinders).
    Others are clean or have minor issues.
    The agent must identify the faulty batch.

    Returns:
        dict with:
            batches         : List of batch dicts (each = list of cylinders)
            faulty_batch_id : str — the ground truth answer
            incident_report : str — field complaint text shown to agent
            affected_zone   : str — zone where complaints came from
    """
    rng = _rng(seed)

    num_batches = rng.randint(3, 5)
    batch_ids = [f"BATCH-20241103-S{i+1:02d}" for i in range(num_batches)]
    faulty_idx = rng.randint(0, num_batches - 1)
    faulty_batch_id = batch_ids[faulty_idx]
    affected_zone = rng.choice(ZONES)

    batches = []
    for i, bid in enumerate(batch_ids):
        if i == faulty_idx:
            # Faulty batch — forced bad profiles
            size = rng.randint(8, 12)
            cylinders = []
            for j in range(size):
                # Mix: some faulty, some okay
                if j < size // 3:
                    profile = rng.choice(["VALVE_LEAK", "RUST_CRITICAL", "MULTIPLE_ISSUES"])
                else:
                    profile = "PERFECT"
                cyl_seed = rng.randint(0, 999999)
                cyl = generate_cylinder(
                    profile_name=profile,
                    seed=cyl_seed,
                    cylinder_id=f"CYL-2024-F-{1000 + j}",
                    batch_id=bid,
                )
                cyl["destination_zone"] = affected_zone
                cylinders.append(cyl)
            batches.append({"batch_id": bid, "cylinders": cylinders})
        else:
            # Clean batch — all perfect
            size = rng.randint(8, 12)
            cylinders = []
            for j in range(size):
                profile = rng.choice(["PERFECT", "BAD_QR"])  # Minor issues only
                cyl_seed = rng.randint(0, 999999)
                cyl = generate_cylinder(
                    profile_name=profile,
                    seed=cyl_seed,
                    cylinder_id=f"CYL-2024-{chr(65+i)}-{1000+j}",
                    batch_id=bid,
                )
                cylinders.append(cyl)
            batches.append({"batch_id": bid, "cylinders": cylinders})

    # Field incident report text shown to agent
    incident_report = (
        f"FIELD INCIDENT REPORT\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Zone: {affected_zone}\n"
        f"Complaints: {rng.randint(2, 6)} customers reported gas smell / suspected leakage.\n"
        f"Cylinders involved: Recently delivered LPG cylinders from batches: "
        f"{', '.join(batch_ids)}\n"
        f"Action required: Identify faulty batch, determine recall scope, "
        f"recommend corrective action."
    )

    return {
        "batches":          batches,
        "faulty_batch_id":  faulty_batch_id,
        "incident_report":  incident_report,
        "affected_zone":    affected_zone,
        "batch_ids":        batch_ids,
    }