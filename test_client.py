"""
Test client.py imports and instantiation.
Full integration test runs after server is up (Phase 7).
"""
from client import LPGInspectorEnv
from models import LPGInspectorAction

print("=== Client Import Test ===")

# Instantiate client (no connection yet)
env = LPGInspectorEnv(base_url="https://placeholder.hf.space")
print(f"Client instantiated: {type(env).__name__}")

# Test action serialization
action = LPGInspectorAction(
    decision="QUARANTINE",
    reason="Valve pressure critically low. Safety risk.",
    defect_flags=["VALVE_PRESSURE_LOW"],
    priority="URGENT",
)
payload = env._step_payload(action)
print(f"\n_step_payload output:")
for k, v in payload.items():
    print(f"  {k}: {v}")

# Test state parsing
state = env._parse_state({
    "episode_id":      "test-123",
    "step_count":      3,
    "task_name":       "batch_inspection",
    "difficulty":      "medium",
    "max_steps":       10,
    "current_score":   0.72,
    "cylinders_total": 10,
    "cylinders_done":  3,
    "batch_id":        "BATCH-20260407-S01",
    "safety_misses":   0,
})
print(f"\n_parse_state output:")
print(f"  episode_id:     {state.episode_id}")
print(f"  step_count:     {state.step_count}")
print(f"  task_name:      {state.task_name}")
print(f"  current_score:  {state.current_score}")
print(f"  cylinders_done: {state.cylinders_done}")

# Test result parsing
result = env._parse_result({
    "done":   False,
    "reward": 0.85,
    "observation": {
        "cylinder_id":        "CYL-2026-A-0042",
        "batch_id":           "BATCH-20260407-S01",
        "weight_kg":          14.228,
        "valve_pressure_bar": 4.06,
        "qr_status":          "VALID",
        "body_condition":     "GOOD",
        "fill_date":          "2026-04-06",
        "previous_failures":  1,
        "destination_zone":   "ZONE_A_HIGH_DEMAND",
        "inspector_note":     "Pressure sensor alert: valve pressure drop detected.",
        "feedback_message":   "Step 1/10. Submit your decision.",
        "progress_score":     0.1,
        "task_name":          "batch_inspection",
        "step_number":        1,
        "total_steps":        10,
        "batch_context": {
            "batch_size":           10,
            "cylinders_processed":  1,
            "cylinders_passed":     0,
            "cylinders_failed":     0,
            "cylinders_quarantine": 0,
            "cylinders_retest":     0,
            "batch_pass_rate":      0.0,
            "batch_alerts":         [],
        },
    },
})
print(f"\n_parse_result output:")
print(f"  observation.cylinder_id:        {result.observation.cylinder_id}")
print(f"  observation.valve_pressure_bar: {result.observation.valve_pressure_bar}")
print(f"  observation.batch_context.batch_size: {result.observation.batch_context.batch_size}")
print(f"  reward:                         {result.reward}")
print(f"  done:                           {result.done}")

print("\nClient tests passed")