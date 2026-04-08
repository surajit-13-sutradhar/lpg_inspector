from server.data_generator import generate_cylinder, generate_batch, generate_incident_scenario
from server.graders import SingleCylinderGrader, BatchInspectionGrader, IncidentRootCauseGrader

print("=" * 60)
print("GRADER 1: SingleCylinderGrader")
print("=" * 60)

grader1 = SingleCylinderGrader()
cyl = generate_cylinder("VALVE_LEAK", seed=42)
print(f"Cylinder profile: VALVE_LEAK")
print(f"Ground truth: {cyl['_ground_truth']} | Expected flags: {cyl['_expected_flags']}")
print()

# Perfect answer
score = grader1.grade(
    decisions=[{"decision": "QUARANTINE", "reason": "Valve pressure critically low.", "defect_flags": ["VALVE_PRESSURE_LOW"], "priority": "URGENT"}],
    cylinder=cyl,
)
print(f"Perfect answer score:          {score:.4f}  (expect ~1.00)")

# Safety miss — most dangerous
score = grader1.grade(
    decisions=[{"decision": "PASS", "reason": "Looks okay.", "defect_flags": [], "priority": "NORMAL"}],
    cylinder=cyl,
)
print(f"Safety miss (PASS) score:      {score:.4f}  (expect ~0.00)")

# Cautious but wrong decision
score = grader1.grade(
    decisions=[{"decision": "RETEST", "reason": "Not sure.", "defect_flags": ["VALVE_PRESSURE_LOW"], "priority": "URGENT"}],
    cylinder=cyl,
)
print(f"RETEST instead of QUARANTINE:  {score:.4f}  (expect ~0.30-0.50)")

# Right decision, wrong flags
score = grader1.grade(
    decisions=[{"decision": "QUARANTINE", "reason": "Something wrong.", "defect_flags": [], "priority": "URGENT"}],
    cylinder=cyl,
)
print(f"Right decision, no flags:      {score:.4f}  (expect ~0.60-0.75)")

print()
print("=" * 60)
print("GRADER 2: BatchInspectionGrader")
print("=" * 60)

grader2 = BatchInspectionGrader()
batch = generate_batch(size=5, difficulty="medium", seed=99)
print(f"Batch of {len(batch)} cylinders:")
for c in batch:
    print(f"  {c['cylinder_id']} | {c['profile_name']:20s} | GT: {c['_ground_truth']}")
print()

# Perfect decisions for each cylinder
perfect_decisions = [
    {"decision": c["_ground_truth"], "defect_flags": c["_expected_flags"], "priority": c["_expected_priority"], "reason": "Correct."}
    for c in batch
]
score = grader2.grade(decisions=perfect_decisions, cylinders=batch)
print(f"All perfect decisions:         {score:.4f}  (expect ~0.90-1.00)")

# All PASS decisions (worst case)
all_pass = [
    {"decision": "PASS", "defect_flags": [], "priority": "NORMAL", "reason": "Looks fine."}
    for _ in batch
]
score = grader2.grade(decisions=all_pass, cylinders=batch)
print(f"All PASS decisions:            {score:.4f}  (expect ~0.00-0.15)")

print()
print("=" * 60)
print("GRADER 3: IncidentRootCauseGrader")
print("=" * 60)

grader3 = IncidentRootCauseGrader()
scenario = generate_incident_scenario(seed=7)
print(f"Faulty batch: {scenario['faulty_batch_id']}")
print(f"All batches:  {scenario['batch_ids']}")
print()

# Perfect answer
score = grader3.grade(
    decisions=[{
        "decision":     scenario["faulty_batch_id"],
        "defect_flags": [scenario["faulty_batch_id"]],
        "reason":       "Valve leak detected in faulty batch. Recommend immediate recall and quarantine.",
        "priority":     "URGENT",
    }],
    scenario=scenario,
)
print(f"Perfect answer:                {score:.4f}  (expect ~0.90-1.00)")

# Wrong batch identified
wrong_batch = [b for b in scenario["batch_ids"] if b != scenario["faulty_batch_id"]][0]
score = grader3.grade(
    decisions=[{
        "decision":     wrong_batch,
        "defect_flags": [wrong_batch],
        "reason":       "This batch seems problematic.",
        "priority":     "URGENT",
    }],
    scenario=scenario,
)
print(f"Wrong batch identified:        {score:.4f}  (expect ~0.10-0.20)")

# Right batch, too wide recall
score = grader3.grade(
    decisions=[{
        "decision":     scenario["faulty_batch_id"],
        "defect_flags": scenario["batch_ids"],  # recalled ALL batches
        "reason":       "Valve pressure issues. Recall all batches to be safe.",
        "priority":     "URGENT",
    }],
    scenario=scenario,
)
print(f"Right batch, too wide recall:  {score:.4f}  (expect ~0.50-0.70)")

print()
print("All grader tests complete.")