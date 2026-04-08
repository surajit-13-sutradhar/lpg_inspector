# LPG Inspector — OpenEnv Environment

An AI Quality Control Decision environment for LPG (Liquefied Petroleum Gas)
cylinder inspection pipelines. The agent acts as an automated triage system,
making safety-critical decisions based on sensor readings and inspection reports.

## Environment Description

LPG cylinder distribution is safety-critical infrastructure. Defective cylinders
cause gas leaks, fires, and fatalities. Human inspectors process thousands of
cylinders daily — a task with high cognitive load, fatigue risk, and severe
consequences for errors.

This environment models the **decision layer** of an automated LPG inspection
pipeline. Sensor readings (weight, valve pressure) and vision model outputs
(body condition, QR status) are pre-processed and presented as structured
reports. The agent's role is multi-step quality triage and dispatch
optimization — tasks requiring contextual reasoning across multiple data points.

## Action Space

The agent submits one `LPGInspectorAction` per cylinder:

| Field | Type | Values | Description |
|---|---|---|---|
| `decision` | str | `PASS` `FAIL` `RETEST` `QUARANTINE` | Primary triage decision |
| `reason` | str | free text | Brief justification (1-2 sentences) |
| `defect_flags` | List[str] | see below | Specific defects identified |
| `priority` | str | `NORMAL` `URGENT` `HOLD` | Dispatch urgency |

**Valid defect flags:**
`WEIGHT_LOW`, `WEIGHT_HIGH`, `WEIGHT_BORDERLINE`,
`VALVE_PRESSURE_LOW`, `VALVE_PRESSURE_HIGH`,
`QR_INVALID`, `QR_MISSING`, `QR_DUPLICATE`,
`BODY_DAMAGE`, `SAFETY_HAZARD`

## Observation Space

The agent receives one `LPGInspectorObservation` per step:

| Field | Type | Description |
|---|---|---|
| `cylinder_id` | str | Unique cylinder serial number |
| `batch_id` | str | Batch this cylinder belongs to |
| `weight_kg` | float | Fill weight (nominal: 14.2 ± 0.15 kg) |
| `valve_pressure_bar` | float | Valve pressure (safe: 6.5–7.5 bar) |
| `qr_status` | str | `VALID` `INVALID` `MISSING` `DUPLICATE` |
| `body_condition` | str | `GOOD` `MINOR_DENT` `MAJOR_DENT` `RUST_MINOR` `RUST_CRITICAL` |
| `fill_date` | str | Date cylinder was filled |
| `previous_failures` | int | Times this cylinder failed before |
| `destination_zone` | str | Assigned delivery zone |
| `inspector_note` | str | Automated system message |
| `batch_context` | BatchContext | Batch-level stats (medium/hard tasks) |
| `incident_report` | str | Field complaint text (hard task only) |
| `available_batch_ids` | List[str] | Batch IDs to investigate (hard task) |
| `feedback_message` | str | Environment feedback on last action |
| `progress_score` | float | Episode progress 0.0–1.0 |
| `done` | bool | Is episode complete? (inherited) |
| `reward` | float | Last step reward (inherited) |

## Tasks

| Task | Difficulty | Description | Max Steps | Baseline Score |
|---|---|---|---|---|
| `single_cylinder_triage` | Easy | Inspect one cylinder, make PASS/FAIL/RETEST/QUARANTINE decision | 5 | 0.990 |
| `batch_inspection` | Medium | Process 10 cylinders, classify each + optimize dispatch priority | 10 | 0.920 |
| `incident_root_cause` | Hard | Investigate field incident, identify faulty batch, determine recall scope | 15 | 0.320 |

## Reward Function

Rewards are **dense** — signal is provided throughout the trajectory, not just at episode end.

**Per-step reward components:**
- `+0.60` core decision correctness
- `+0.20` defect flag precision/recall (F1)
- `+0.15` priority assignment correctness
- `+0.05` efficiency bonus
- `-0.40` safety miss penalty (FAIL/QUARANTINE labeled as PASS)

**Safety miss** (approving a dangerous cylinder) incurs a heavy penalty.
This models real-world consequences where missing a defect is far worse
than being overcautious.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ws` | WebSocket | Primary training endpoint (persistent) |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action |
| `/state` | GET | Episode metadata |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## Setup & Usage

### Local Development

```bash
# Install
pip install openenv-core fastapi uvicorn pydantic openai python-dotenv

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health
```

### Docker

```bash
docker build -t lpg-inspector:latest -f server/Dockerfile .
docker run -p 8000:8000 lpg-inspector:latest
```

### Python Client

```python
from client import LPGInspectorEnv
from models import LPGInspectorAction

with LPGInspectorEnv(base_url="https://YOUR_USERNAME-lpg-inspector.hf.space").sync() as env:
    result = env.reset(task_name="single_cylinder_triage")
    print(result.observation.cylinder_id)
    print(result.observation.weight_kg)

    result = env.step(LPGInspectorAction(
        decision="QUARANTINE",
        reason="Valve pressure below safe threshold.",
        defect_flags=["VALVE_PRESSURE_LOW"],
        priority="URGENT",
    ))
    print(result.reward)
    print(result.done)
```

### Running the Baseline

```bash
# Set environment variables
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LPG_ENV_URL=https://YOUR_USERNAME-lpg-inspector.hf.space

python inference.py
```

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Providers:

| Task | Score | Steps | Status |
|---|---|---|---|
| `single_cylinder_triage` | 0.990 | 1 | ✅ PASS |
| `batch_inspection` | 0.920 | 10 | ✅ PASS |
| `incident_root_cause` | 0.320 | 1 | ❌ FAIL |
| **Average** | **0.743** | — | — |

The hard task (incident root cause) correctly identifies the faulty batch
but tends to over-recall, including clean batches in the recall scope.
This is expected baseline behavior — a trained agent should learn to
narrow the recall scope precisely.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace API token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `LPG_ENV_URL` | `http://localhost:8000` | Environment URL |

## Cylinder Profiles

The environment generates synthetic cylinders from 12 profiles:

| Profile | Ground Truth | Key Defect |
|---|---|---|
| PERFECT | PASS | None |
| UNDERWEIGHT | FAIL | WEIGHT_LOW |
| OVERWEIGHT | FAIL | WEIGHT_HIGH |
| VALVE_LEAK | QUARANTINE | VALVE_PRESSURE_LOW |
| VALVE_OVERPRESSURE | QUARANTINE | VALVE_PRESSURE_HIGH |
| BAD_QR | RETEST | QR_INVALID |
| MISSING_QR | RETEST | QR_MISSING |
| DUPLICATE_QR | QUARANTINE | QR_DUPLICATE |
| MAJOR_DENT | FAIL | BODY_DAMAGE |
| RUST_CRITICAL | QUARANTINE | SAFETY_HAZARD |
| BORDERLINE_WEIGHT | RETEST | WEIGHT_BORDERLINE |
| MULTIPLE_ISSUES | QUARANTINE | Multiple |

## License

MIT