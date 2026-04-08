# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os
import io
from huggingface_hub import HfApi

api = HfApi()
token = os.getenv("HF_TOKEN")

# Write README with proper frontmatter
# Using unicode escape for emoji to avoid Windows encoding issues
fire_emoji = "\U0001f525"  # fire emoji

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

## Tasks

| Task | Difficulty | Max Steps | Baseline Score |
|---|---|---|---|
| single_cylinder_triage | Easy | 5 | 0.990 |
| batch_inspection | Medium | 10 | 0.920 |
| incident_root_cause | Hard | 15 | 0.320 |

## API Endpoints

- POST /reset - start new episode
- POST /step  - submit action
- GET  /state - episode metadata
- GET  /health - health check

## Baseline Scores

| Task | Score | Status |
|---|---|---|
| single_cylinder_triage | 0.990 | PASS |
| batch_inspection | 0.920 | PASS |
| incident_root_cause | 0.320 | FAIL |
| Average | 0.743 | - |
""".encode("utf-8")

api.upload_file(
    path_or_fileobj = io.BytesIO(readme_content),
    path_in_repo    = "README.md",
    repo_id         = "crow1234des/lpg-inspector",
    repo_type       = "space",
    token           = token,
)
print("README.md updated successfully.")