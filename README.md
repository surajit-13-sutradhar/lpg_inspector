# 🔥 LPG Inspector - OpenEnv Environment


[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/crow1234des/lpg-inspector)
[![OpenEnv Validated](https://img.shields.io/badge/OpenEnv-Validated%20✓-green)](https://huggingface.co/spaces/crow1234des/lpg-inspector)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What Is This?

**LPG Inspector** is a production-grade reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates the **decision-making layer of an industrial LPG cylinder quality control pipeline** - a real-world, safety-critical task where errors can cause gas leaks, fires, and fatalities.

An AI agent acts as an automated Quality Control Inspector. It receives structured sensor readings and inspection reports - weight, valve pressure, QR scan status, body condition - and must make triage decisions: approve a cylinder for dispatch, reject it, send it for retest, or quarantine it immediately.

This environment was designed, built, and deployed as a hackathon submission and **cleared Phase 1 automated validation** and **advanced to Round 2** of the Meta PyTorch x Scaler School of Technology OpenEnv competition.

---

## The Problem It Solves

Hindustan Petroleum (HP) and similar LPG distributors process **millions of cylinders daily**. Each cylinder must be checked for:

- Correct fill weight (underfilled = dangerous, overfilled = explosive)
- Safe valve pressure (leaks cause fires)
- Valid QR tagging (duplicates = supply chain fraud)
- Body integrity (dents and rust compromise structural safety)

Human inspectors doing this under time pressure and fatigue make errors. This environment provides a training and evaluation ground for AI agents that can learn to perform this inspection reliably - with a reward function that **heavily penalizes safety misses**, modeling real-world consequences.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  TRAINING / INFERENCE CODE                              │
│                                                         │
│  env = LPGInspectorEnv(base_url="https://...")          │
│  result = env.reset(task_name="batch_inspection")       │
│  result = env.step(LPGInspectorAction(...))             │
│                                                         │
└──────────────────────┬──────────────────────────────────┘
                       │  WebSocket /ws  (persistent)
                       │  HTTP /reset /step /state /health
                       │
┌──────────────────────▼──────────────────────────────────┐
│  DOCKER CONTAINER  (HuggingFace Spaces · Free Tier)     │
│                                                         │
│  FastAPI Server                                         │
│  └─ LPGInspectorEnvironment                             │
│      ├─ reset()  → generates cylinder scenario          │
│      ├─ step()   → evaluates decision, returns reward   │
│      └─ state    → episode metadata                     │
│                                                         │
│  Synthetic Data Generator (12 cylinder profiles)        │
│  Deterministic Graders (3 tasks, 0.0 -1.0 scoring)      │
└─────────────────────────────────────────────────────────┘
```

Built on the OpenEnv microservice philosophy: **environments as containers, not in-process libraries.**

---

## Three Tasks, Three Difficulty Levels

### Task 1 - `single_cylinder_triage` · Easy
**Objective:** Inspect one cylinder. Make the correct PASS / FAIL / RETEST / QUARANTINE decision based on sensor readings.

- One cylinder per episode, up to 5 steps
- Clear-cut cases: obvious weight violations, pressure anomalies, QR failures
- Tests: basic threshold reasoning
- **Baseline score: 0.990**

---

### Task 2 - `batch_inspection` · Medium
**Objective:** Process a full batch of 10 cylinders. Classify each correctly and flag urgent dispatch priorities.

- 10 cylinders per episode, one per step
- Mix of obvious and borderline cases
- Tests: sustained attention, batch-level context tracking
- **Baseline score: 0.706**

---

### Task 3 - `incident_root_cause` · Hard
**Objective:** A field incident has been reported - gas leak complaints from a delivery zone. The agent receives dispatch logs and inspection records across 3 -5 batches. Identify the faulty batch, determine the correct recall scope, and recommend corrective action.

- Multi-document reasoning across batches
- Red herrings: clean batches with superficially similar profiles
- Tests: evidence chaining, causal reasoning, precision recall (not over-recall)
- **Baseline score: 0.060** (correctly identifies faulty batch but over-recalls - as expected for an untrained baseline)

---

## Reward Function Design

The reward function is **dense** - signal is provided throughout every episode, not just at termination. This is intentional: sparse rewards give agents almost no gradient to learn from.

```
Per-step reward breakdown:

  +0.60  Core decision correctness
  +0.20  Defect flag accuracy (F1 - precision + recall on identified defects)
  +0.15  Priority assignment (URGENT/NORMAL/HOLD)
  +0.05  Efficiency bonus (fewer steps = small reward)

   -0.40  Safety miss penalty (FAIL/QUARANTINE labeled as PASS)
   -0.10  Unnecessary RETEST on clean cylinder
```

The ** -0.40 safety miss penalty** is the key design decision. It models the asymmetry of real-world consequences: approving a dangerous cylinder is catastrophically worse than being overcautious. This creates the right incentive structure for an agent learning safety-critical inspection.

---

## Action & Observation Spaces

### Action: `LPGInspectorAction`

```python
class LPGInspectorAction(Action):
    decision:     str        # "PASS" | "FAIL" | "RETEST" | "QUARANTINE"
    reason:       str        # Natural language justification
    defect_flags: List[str]  # ["WEIGHT_LOW", "VALVE_PRESSURE_LOW", ...]
    priority:     str        # "NORMAL" | "URGENT" | "HOLD"
```

### Observation: `LPGInspectorObservation`

```python
class LPGInspectorObservation(Observation):
    # Cylinder identification
    cylinder_id:          str    # e.g. "CYL-2026-A-0042"
    batch_id:             str    # e.g. "BATCH-20260407-S01"

    # Sensor readings
    weight_kg:            float  # nominal: 14.2 ± 0.15 kg
    valve_pressure_bar:   float  # safe range: 6.5  - 7.5 bar
    qr_status:            str    # "VALID" | "INVALID" | "MISSING" | "DUPLICATE"
    body_condition:       str    # "GOOD" | "MINOR_DENT" | "MAJOR_DENT" | ...

    # Context
    fill_date:            str
    previous_failures:    int
    destination_zone:     str
    inspector_note:       str    # automated system message

    # Batch context (medium/hard tasks)
    batch_context:        Optional[BatchContext]

    # Incident context (hard task only)
    incident_report:      Optional[str]
    available_batch_ids:  List[str]

    # Episode feedback (inherited: done, reward)
    feedback_message:     str
    progress_score:       float
```

---

## Synthetic Data Generator

All cylinder data is procedurally generated - no external datasets, no privacy concerns, fully reproducible with seeds.

**12 cylinder profiles**, ranging from trivially correct to dangerously ambiguous:

| Profile | Ground Truth | Primary Defect |
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
| MULTIPLE_ISSUES | QUARANTINE | WEIGHT_LOW + VALVE_PRESSURE_LOW + BODY_DAMAGE |

---

## Baseline Scores

Evaluated against the live HuggingFace Space using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Providers:

| Task | Difficulty | Score | Steps | Status |
|---|---|---|---|---|
| `single_cylinder_triage` | Easy | **0.990** | 1 |   PASS |
| `batch_inspection` | Medium | **0.706** | 10 |   PASS |
| `incident_root_cause` | Hard | **0.060** | 1 |  FAIL |
| **Average** | - | **0.585** | - | - |

The hard task correctly identifies the faulty batch but over-recalls (includes clean batches in the recall scope). This is expected baseline behavior - a trained agent with GRPO should learn precise recall scoping.

---

## Project Structure

```
lpg_inspector/
├── inference.py              ← Baseline script (OpenAI client, [START][STEP][END] logs)
├── openenv.yaml              ← OpenEnv manifest (validated)
├── README.md
├── pyproject.toml            ← Package with [project.scripts] entry point
├── requirements.txt
├── uv.lock
│
├── models.py                 ← Pydantic type contracts (Action, Observation, State)
├── client.py                 ← WebSocket/HTTP client (EnvClient subclass)
│
└── server/
    ├── __init__.py
    ├── app.py                ← FastAPI wiring (create_fastapi_app)
    ├── environment.py        ← Core logic: reset(), step(), state
    ├── data_generator.py     ← Synthetic cylinder data (12 profiles, seeded)
    ├── graders.py            ← Deterministic graders for all 3 tasks
    └── Dockerfile            ← python:3.11-slim, port 7860
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check - returns `{"status":"healthy"}` |
| `/reset` | POST | Start new episode, returns first observation |
| `/step` | POST | Submit action, returns next observation + reward |
| `/state` | GET | Episode metadata (step count, score, safety misses) |
| `/ws` | WebSocket | Persistent connection for training loops |
| `/docs` | GET | Interactive Swagger UI |

---

## Setup & Usage

### Prerequisites
```
Python 3.10+
Docker (for containerized deployment)
HuggingFace account + token
```

### Local Development

```bash
# Clone and install
git clone https://huggingface.co/spaces/crow1234des/lpg-inspector
cd lpg-inspector
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health
# {"status":"healthy"}
```

### Docker

```bash
docker build -t lpg-inspector:latest -f server/Dockerfile .
docker run -p 7860:7860 lpg-inspector:latest
```

### Python Client

```python
from client import LPGInspectorEnv
from models import LPGInspectorAction

with LPGInspectorEnv(base_url="https://crow1234des-lpg-inspector.hf.space").sync() as env:

    # Easy task - single cylinder
    result = env.reset(task_name="single_cylinder_triage")
    print(f"Cylinder: {result.observation.cylinder_id}")
    print(f"Weight:   {result.observation.weight_kg} kg")
    print(f"Pressure: {result.observation.valve_pressure_bar} bar")

    result = env.step(LPGInspectorAction(
        decision     = "QUARANTINE",
        reason       = "Valve pressure below safe threshold - leak risk.",
        defect_flags = ["VALVE_PRESSURE_LOW"],
        priority     = "URGENT",
    ))
    print(f"Reward:   {result.reward}")
    print(f"Feedback: {result.observation.feedback_message}")
```

### Running the Baseline

```bash
# Set environment variables
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LPG_ENV_URL=https://crow1234des-lpg-inspector.hf.space

python inference.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | - | HuggingFace API token (required) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM inference endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `LPG_ENV_URL` | `https://crow1234des-lpg-inspector.hf.space` | Environment URL |

---

## Technical Stack

| Component | Technology |
|---|---|
| Environment framework | OpenEnv (Meta PyTorch) |
| API server | FastAPI + Uvicorn |
| Type contracts | Pydantic v2 |
| Transport | WebSocket (persistent) + HTTP |
| Container | Docker · python:3.11-slim |
| Deployment | HuggingFace Spaces (Docker SDK) |
| Baseline LLM | Qwen/Qwen2.5-72B-Instruct |
| LLM client | OpenAI Python SDK |

---

## Design Decisions Worth Noting

**Why synthetic data?**
Real cylinder inspection data from HP would contain PII and proprietary operational data. Synthetic generation gives us full control over difficulty distribution, reproducibility, and ground truth labels - all required for deterministic grading.

**Why WebSocket over HTTP?**
HTTP requires a new TCP handshake per step - 10 -50ms overhead. WebSocket maintains a persistent connection - ~0.1ms per step. In a training loop running thousands of steps, this is the difference between a fast feedback loop and an unusably slow one.

**Why the  -0.40 safety miss penalty?**
In industrial safety inspection, the cost of a false negative (approving a dangerous cylinder) is orders of magnitude higher than a false positive (rejecting a good one). The reward function encodes this asymmetry directly, giving agents the right incentive to be conservative on borderline cases.

**Why three difficulty levels?**
The easy task establishes a floor - any agent that can reason about thresholds should score near 1.0. The medium task requires multi-step context tracking. The hard task requires evidence chaining across multiple documents and deliberate recall scoping. This progression creates a meaningful evaluation ladder.

---

## Hackathon Context

This project was built for the **Meta PyTorch x Scaler School of Technology OpenEnv Hackathon, Round 1**.

The competition required participants to build a real-world OpenEnv environment with:
- Full OpenEnv spec compliance (typed models, step/reset/state API, openenv.yaml)
- Minimum 3 tasks with deterministic graders (0.0 -1.0)
- Dense reward function with partial progress signals
- Baseline inference script with exact structured logging
- Deployment to HuggingFace Spaces with working Dockerfile

**This submission passed all 4 Phase 1 automated gates:**
-   OpenEnv Reset (POST OK)
-   Dockerfile at repo root
-   inference.py at repo root
-   openenv validate

**And advanced to Round 2** (Phase 2: Agentic Evaluation).

---

## What I Learned Building This

- **OpenEnv architecture** - environments as containerized microservices, WebSocket vs HTTP tradeoffs
- **Reward shaping** - why dense rewards matter for RL training, how to encode domain-specific cost asymmetries
- **FastAPI + Pydantic** - type-safe API design, automatic serialization, Swagger generation
- **Docker deployment** - layer caching, port mapping, HuggingFace Spaces constraints
- **RL environment design** - episode boundaries, state management, deterministic grading
- **Production debugging** - WebSocket disconnect handling, retry logic, encoding issues across platforms

---

## License

MIT - feel free to fork, extend, or use as a reference for building your own OpenEnv environments.

---

## Links

- **Live Space:** https://huggingface.co/spaces/crow1234des/lpg-inspector
- **API Endpoint:** https://crow1234des-lpg-inspector.hf.space