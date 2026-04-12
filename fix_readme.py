# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os
import io
from huggingface_hub import HfApi

api = HfApi()
token = os.getenv("HF_TOKEN")

fire_emoji = "\U0001f525"

readme_content = f"""---
title: LPG Inspector
emoji: {fire_emoji}
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - industrial
  - safety
  - quality-control
  - lpg
---

# LPG Inspector - OpenEnv Environment

An AI Quality Control Decision environment for LPG (Liquefied Petroleum Gas)
cylinder inspection pipelines. The agent acts as an automated triage system,
making safety-critical decisions based on sensor readings and inspection reports.

## Environment Description

LPG cylinder distribution is safety-critical infrastructure. Defective cylinders
cause gas leaks, fires, and fatalities. Human inspectors process thousands of
cylinders daily -- a task with high cognitive load, fatigue risk, and severe
consequences for errors.

This environment models the decision layer of an automated LPG inspection
pipeline. Sensor readings (weight, valve pressure) and vision model outputs
(body condition, QR status) are pre-processed and presented as structured
reports. The agent makes triage and dispatch decisions requiring contextual
reasoning across multiple data points.

## Action Space

The agent submits one LPGInspectorAction per cylinder:

- decision: PASS | FAIL | RETEST | QUARANTINE
- reason: brief natural language justification
- defect_flags: list of detected defects (WEIGHT_LOW, VALVE_PRESSURE_LOW, etc.)
- priority: NORMAL | URGENT | HOLD

## Observation Space

Each step the agent receives:

- cylinder_id, batch_id
- weight_kg (nominal: 14.2 +/- 0.15 kg)
- valve_pressure_bar (safe: 6.5 to 7.5 bar)
- qr_status: VALID | INVALID | MISSING | DUPLICATE
- body_condition: GOOD | MINOR_DENT | MAJOR_DENT | RUST_MINOR | RUST_CRITICAL
- fill_date, previous_failures, destination_zone
- inspector_note, feedback_message, progress_score
- batch_context (medium/hard tasks)
- incident_report, available_batch_ids (hard task)

## Tasks

| Task | Difficulty | Max Steps | Baseline Score |
|---|---|---|---|
| single_cylinder_triage | Easy | 5 | 0.990 |
| batch_inspection | Medium | 10 | 0.706 |
| incident_root_cause | Hard | 15 | 0.060 |

## Reward Function

Dense rewards throughout trajectory:
- +0.60 core decision correctness
- +0.20 defect flag accuracy (F1)
- +0.15 priority assignment correctness
- +0.05 efficiency bonus
- -0.40 safety miss penalty (FAIL/QUARANTINE labeled as PASS)

## API Endpoints

- POST /reset - start new episode
- POST /step  - submit action
- GET  /state - episode metadata
- GET  /health - health check
- GET  /docs  - Swagger UI

## Setup

```bash
pip install openenv-core fastapi uvicorn pydantic openai python-dotenv
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Environment Variables

- HF_TOKEN - HuggingFace API token
- API_BASE_URL - LLM endpoint (default: https://router.huggingface.co/v1)
- MODEL_NAME - model identifier (default: Qwen/Qwen2.5-72B-Instruct)
- LPG_ENV_URL - environment URL

## Baseline Scores

Evaluated using Qwen/Qwen2.5-72B-Instruct against live HF Space:

| Task | Score | Steps | Status |
|---|---|---|---|
| single_cylinder_triage | 0.990 | 1 | PASS |
| batch_inspection | 0.706 | 10 | PASS |
| incident_root_cause | 0.060 | 1 | FAIL |
| Average | 0.585 | - | - |

The hard task correctly identifies the faulty batch but over-recalls,
including clean batches in the recall scope. This is expected baseline
behavior -- a trained agent should learn precise recall scoping.

## Cylinder Profiles

12 synthetic profiles from PERFECT to MULTIPLE_ISSUES covering all
real-world LPG defect categories: weight violations, valve pressure
anomalies, QR code issues, body damage, and combined failures.

## License

MIT
""".encode("utf-8")

api.upload_file(
    path_or_fileobj = io.BytesIO(readme_content),
    path_in_repo    = "README.md",
    repo_id         = "crow1234des/lpg-inspector",
    repo_type       = "space",
    token           = token,
)
print("README updated with live baseline scores.")