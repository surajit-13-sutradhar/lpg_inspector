import sys
sys.path.insert(0, ".")

from server.environment import LPGInspectorEnvironment
from models import LPGInspectorAction

print("=" * 60)
print("TASK 1: single_cylinder_triage")
print("=" * 60)

env = LPGInspectorEnvironment()

obs = env.reset(seed=42, task_name="single_cylinder_triage")
print(f"Reset OK")
print(f"  cylinder_id:        {obs.cylinder_id}")
print(f"  weight_kg:          {obs.weight_kg}")
print(f"  valve_pressure_bar: {obs.valve_pressure_bar}")
print(f"  qr_status:          {obs.qr_status}")
print(f"  body_condition:     {obs.body_condition}")
print(f"  inspector_note:     {obs.inspector_note}")
print(f"  done:               {obs.done}")
print(f"  task_name:          {obs.task_name}")
print()

action = LPGInspectorAction(
    decision="PASS",
    reason="Looks fine.",
    defect_flags=[],
    priority="NORMAL",
)
obs = env.step(action)
print(f"Step result:")
print(f"  done:     {obs.done}")
print(f"  reward:   {obs.reward}")
print(f"  feedback: {obs.feedback_message}")
print(f"  score:    {env.state.current_score}")
print()

print("=" * 60)
print("TASK 2: batch_inspection")
print("=" * 60)

obs = env.reset(seed=42, task_name="batch_inspection")
print(f"Reset OK — {obs.batch_context.batch_size} cylinders in batch")
print(f"  First cylinder: {obs.cylinder_id} | {obs.inspector_note}")
print()

# Process all 10 cylinders with QUARANTINE for everything
for i in range(10):
    if obs.done:
        break
    action = LPGInspectorAction(
        decision="QUARANTINE",
        reason="Being cautious.",
        defect_flags=["SAFETY_HAZARD"],
        priority="URGENT",
    )
    obs = env.step(action)
    print(f"  Step {i+1}: done={obs.done} reward={obs.reward:.4f} progress={obs.progress_score:.2f}")

print(f"Final score: {env.state.current_score:.4f}")
print()

print("=" * 60)
print("TASK 3: incident_root_cause")
print("=" * 60)

obs = env.reset(seed=7, task_name="incident_root_cause")
print(f"Reset OK")
print(f"  Available batches: {obs.available_batch_ids}")
print(f"  Incident report preview: {obs.incident_report[:100]}...")
print()

# Submit final answer on first step
action = LPGInspectorAction(
    decision=obs.available_batch_ids[1],  # guess second batch
    reason="Valve leak and rust detected. Recommend immediate recall.",
    defect_flags=[obs.available_batch_ids[1]],
    priority="URGENT",
)
obs = env.step(action)
print(f"  done:     {obs.done}")
print(f"  reward:   {obs.reward}")
print(f"  feedback: {obs.feedback_message}")
print()

print("State after episode:")
state = env.state
print(f"  episode_id:    {state.episode_id[:8]}...")
print(f"  step_count:    {state.step_count}")
print(f"  task_name:     {state.task_name}")
print(f"  current_score: {state.current_score}")
print(f"  safety_misses: {state.safety_misses}")

print()
print("All environment tests complete.")