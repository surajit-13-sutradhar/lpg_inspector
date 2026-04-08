from models import LPGInspectorAction, LPGInspectorObservation, LPGInspectorState, BatchContext

# Test Action
action = LPGInspectorAction(
    decision="QUARANTINE",
    reason="Valve pressure critically low, safety risk.",
    defect_flags=["VALVE_PRESSURE_LOW"],
    priority="URGENT",
)
print("=== Action ===")
print(f"  decision:     {action.decision}")
print(f"  reason:       {action.reason}")
print(f"  defect_flags: {action.defect_flags}")
print(f"  priority:     {action.priority}")

# Test Observation
obs = LPGInspectorObservation(
    done=False,
    reward=None,
    cylinder_id="CYL-2026-A-0042",
    batch_id="BATCH-20260406-S01",
    weight_kg=14.228,
    valve_pressure_bar=4.06,
    qr_status="VALID",
    body_condition="GOOD",
    fill_date="2026-04-05",
    previous_failures=1,
    destination_zone="ZONE_A_HIGH_DEMAND",
    inspector_note="Pressure sensor alert: valve pressure drop detected.",
    feedback_message="Episode started. Inspect the cylinder.",
    progress_score=0.0,
    task_name="single_cylinder_triage",
    step_number=1,
    total_steps=5,
)
print("\n=== Observation ===")
print(f"  cylinder_id:        {obs.cylinder_id}")
print(f"  weight_kg:          {obs.weight_kg}")
print(f"  valve_pressure_bar: {obs.valve_pressure_bar}")
print(f"  qr_status:          {obs.qr_status}")
print(f"  done:               {obs.done}")
print(f"  reward:             {obs.reward}")

# Test BatchContext
ctx = BatchContext(
    batch_size=10,
    cylinders_processed=3,
    cylinders_passed=2,
    cylinders_failed=1,
    batch_pass_rate=0.67,
    batch_alerts=["VALVE_PRESSURE_LOW on CYL-2026-A-1001"],
)
print("\n=== BatchContext ===")
print(f"  batch_size:          {ctx.batch_size}")
print(f"  cylinders_processed: {ctx.cylinders_processed}")
print(f"  batch_pass_rate:     {ctx.batch_pass_rate}")

# Test State
state = LPGInspectorState(
    episode_id="test-episode-001",
    step_count=1,
    task_name="single_cylinder_triage",
    difficulty="easy",
    max_steps=5,
    current_score=0.0,
    cylinders_total=1,
    cylinders_done=0,
)
print("\n=== State ===")
print(f"  episode_id:      {state.episode_id}")
print(f"  step_count:      {state.step_count}")
print(f"  task_name:       {state.task_name}")
print(f"  difficulty:      {state.difficulty}")
print(f"  max_steps:       {state.max_steps}")

print("\nAll models instantiate correctly")