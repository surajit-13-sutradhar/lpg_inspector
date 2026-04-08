from server.data_generator import generate_cylinder, generate_batch, generate_incident_scenario

print("=== Single Cylinder (PERFECT) ===")
cyl = generate_cylinder("PERFECT", seed=42)
for k, v in cyl.items():
    print(f"  {k}: {v}")

print()
print("=== Single Cylinder (VALVE_LEAK) ===")
cyl = generate_cylinder("VALVE_LEAK", seed=42)
for k, v in cyl.items():
    print(f"  {k}: {v}")

print()
print("=== Batch of 5 (medium) ===")
batch = generate_batch(size=5, difficulty="medium", seed=99)
for c in batch:
    print(f"  {c['cylinder_id']} | {c['profile_name']:20s} | GT: {c['_ground_truth']}")

print()
print("=== Incident Scenario ===")
scenario = generate_incident_scenario(seed=7)
print(f"  Faulty batch: {scenario['faulty_batch_id']}")
print(f"  Affected zone: {scenario['affected_zone']}")
print(f"  Num batches: {len(scenario['batches'])}")
print(f"  Incident report preview:")
print(f"  {scenario['incident_report'][:200]}")