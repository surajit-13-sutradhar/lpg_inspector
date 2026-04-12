"""
inference.py
Baseline inference script for the LPG Inspector environment.

MANDATORY REQUIREMENTS:
- Named inference.py, placed in root directory
- Uses OpenAI client for all LLM calls
- Reads credentials from environment variables
- Emits exact [START] / [STEP] / [END] log format
- Completes in under 20 minutes
- Runs on 2 vCPU / 8GB RAM

USAGE:
    set HF_TOKEN=hf_...
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    set LPG_ENV_URL=https://crow1234des-lpg-inspector.hf.space
    python inference.py
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import LPGInspectorEnv
from models import LPGInspectorAction

# ─── Configuration ────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("LPG_ENV_URL",  "https://crow1234des-lpg-inspector.hf.space")
BENCHMARK    = "lpg_inspector"

TEMPERATURE       = 0.2
MAX_TOKENS        = 300
SUCCESS_THRESHOLD = 0.5
MAX_RETRIES       = 3
RETRY_DELAY_S     = 5

MAX_STEPS = {
    "single_cylinder_triage": 5,
    "batch_inspection":       10,
    "incident_root_cause":    15,
}

# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Quality Control Inspector for LPG (Liquefied Petroleum Gas) cylinders.
    You receive sensor readings and inspection reports for cylinders and must make
    safety-critical triage decisions.

    DECISION OPTIONS:
    - PASS       : Cylinder meets all standards. Approve for dispatch.
    - FAIL       : Cylinder fails inspection. Reject and return for investigation.
    - RETEST     : Inconclusive reading. Send for re-inspection.
    - QUARANTINE : Safety risk detected. Isolate immediately. Do NOT dispatch.

    SAFETY THRESHOLDS:
    - Weight:   14.05 to 14.35 kg is acceptable. Outside = FAIL or QUARANTINE.
    - Pressure: 6.5 to 7.5 bar is safe. Below 6.5 = QUARANTINE (valve leak risk).
                Above 7.5 = QUARANTINE (overpressure risk).
    - QR INVALID/MISSING = RETEST
    - QR DUPLICATE = QUARANTINE (identity fraud risk)
    - MAJOR_DENT or RUST_CRITICAL = FAIL or QUARANTINE

    DEFECT FLAGS (use all that apply):
    WEIGHT_LOW, WEIGHT_HIGH, WEIGHT_BORDERLINE,
    VALVE_PRESSURE_LOW, VALVE_PRESSURE_HIGH,
    QR_INVALID, QR_MISSING, QR_DUPLICATE,
    BODY_DAMAGE, SAFETY_HAZARD

    PRIORITY:
    - URGENT : Safety risk or high-demand zone
    - NORMAL : Standard dispatch
    - HOLD   : Pending further review

    RESPONSE FORMAT (strictly follow this):
    DECISION: <PASS|FAIL|RETEST|QUARANTINE>
    FLAGS: <comma-separated defect flags, or NONE>
    PRIORITY: <URGENT|NORMAL|HOLD>
    REASON: <one sentence explanation>
""").strip()

INCIDENT_SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Quality Control Inspector investigating a field incident
    involving LPG cylinder gas leaks.

    You will receive an incident report and a list of batch IDs.
    Your job is to identify which batch is faulty and recommend a recall.

    RESPONSE FORMAT (strictly follow this):
    DECISION: <the faulty batch ID, e.g. BATCH-20241103-S02>
    FLAGS: <comma-separated batch IDs to recall — only the faulty one>
    PRIORITY: URGENT
    REASON: <explain root cause and corrective action in one sentence>
""").strip()


# ─── Mandatory Logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Interaction ──────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str) -> str:
    """Call LLM and return raw text. Never raises — returns empty string on failure."""
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def parse_llm_response(text: str) -> LPGInspectorAction:
    """Parse LLM response into a typed action. Robust — handles imperfect formatting."""
    text = text.strip()

    # Extract DECISION
    decision = "RETEST"
    m = re.search(r"DECISION:\s*([A-Z_\-0-9]+)", text)
    if m:
        decision = m.group(1).strip()

    # Extract FLAGS
    defect_flags = []
    m = re.search(r"FLAGS:\s*(.+?)(?:\n|$)", text)
    if m:
        raw_flags = m.group(1).strip()
        if raw_flags.upper() != "NONE":
            defect_flags = [
                f.strip().upper()
                for f in raw_flags.split(",")
                if f.strip()
            ]

    # Extract PRIORITY
    priority = "NORMAL"
    m = re.search(r"PRIORITY:\s*(URGENT|NORMAL|HOLD)", text)
    if m:
        priority = m.group(1).strip()

    # Extract REASON
    reason = ""
    m = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if m:
        reason = m.group(1).strip()[:500]

    return LPGInspectorAction(
        decision     = decision,
        reason       = reason,
        defect_flags = defect_flags,
        priority     = priority,
    )


# ─── Observation Formatting ───────────────────────────────────────────────────

def format_observation(obs) -> str:
    """Format typed observation into natural language for the LLM."""

    # Incident task — show report
    if obs.incident_report:
        batches_str = "\n".join(f"  - {b}" for b in obs.available_batch_ids)
        return textwrap.dedent(f"""
            {obs.incident_report}

            AVAILABLE BATCH IDs TO INVESTIGATE:
            {batches_str}

            INSTRUCTIONS: Identify ONE faulty batch.
            Set FLAGS to contain ONLY that batch ID.
            {obs.feedback_message}
        """).strip()

    # Single cylinder or batch task
    lines = [
        "CYLINDER INSPECTION REPORT",
        f"Cylinder ID:       {obs.cylinder_id}",
        f"Batch ID:          {obs.batch_id}",
        "",
        "SENSOR READINGS:",
        f"  Weight:          {obs.weight_kg} kg  (acceptable: 14.05 to 14.35 kg)",
        f"  Valve Pressure:  {obs.valve_pressure_bar} bar  (safe: 6.5 to 7.5 bar)",
        f"  QR Status:       {obs.qr_status}",
        f"  Body Condition:  {obs.body_condition}",
        "",
        "CONTEXT:",
        f"  Fill Date:       {obs.fill_date}",
        f"  Prev Failures:   {obs.previous_failures}",
        f"  Dest Zone:       {obs.destination_zone}",
        f"  Inspector Note:  {obs.inspector_note}",
    ]

    if obs.batch_context:
        ctx = obs.batch_context
        lines += [
            "",
            "BATCH PROGRESS:",
            f"  Processed: {ctx.cylinders_processed}/{ctx.batch_size}",
            f"  Passed: {ctx.cylinders_passed} | "
            f"Failed: {ctx.cylinders_failed} | "
            f"Quarantined: {ctx.cylinders_quarantine} | "
            f"Retest: {ctx.cylinders_retest}",
        ]
        if ctx.batch_alerts:
            lines.append(f"  Alerts: {'; '.join(ctx.batch_alerts)}")

    lines += [
        "",
        f"Step {obs.step_number}/{obs.total_steps}",
        obs.feedback_message,
    ]

    return "\n".join(lines)


# ─── Task Runner ──────────────────────────────────────────────────────────────

async def _try_reset(env: LPGInspectorEnv, task_name: str):
    """Attempt reset with retries. Returns result or raises."""
    for attempt in range(MAX_RETRIES):
        try:
            return await env.reset(task_name=task_name)
        except Exception as exc:
            print(f"[DEBUG] reset() attempt {attempt+1}/{MAX_RETRIES} failed: {exc}", flush=True)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_S)
            else:
                raise


async def _try_step(env: LPGInspectorEnv, action: LPGInspectorAction, task_name: str):
    """Attempt step with retries on WebSocket disconnect. Returns (result, error_str)."""
    for attempt in range(MAX_RETRIES):
        try:
            result = await env.step(action)
            return result, None
        except Exception as exc:
            error_msg = str(exc)
            print(f"[DEBUG] step() attempt {attempt+1}/{MAX_RETRIES} failed: {error_msg}", flush=True)

            is_disconnect = any(
                code in error_msg for code in ["1012", "1011", "1006", "service restart", "connect call failed"]
            )

            if is_disconnect and attempt < MAX_RETRIES - 1:
                # Reconnect — create fresh env and reset
                print(f"[DEBUG] Reconnecting...", flush=True)
                try:
                    await env.close()
                except Exception:
                    pass
                await asyncio.sleep(RETRY_DELAY_S)
                try:
                    await env.reset(task_name=task_name)
                except Exception:
                    pass
            else:
                return None, error_msg[:100]

    return None, "max_retries_exceeded"


async def run_task(task_name: str, client: OpenAI) -> dict:
    """Run one full task episode. Always emits [END]. Never raises."""

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    max_steps                = MAX_STEPS[task_name]

    system = (
        INCIDENT_SYSTEM_PROMPT
        if task_name == "incident_root_cause"
        else SYSTEM_PROMPT
    )

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = LPGInspectorEnv(base_url=ENV_URL)

    try:
        # ── Reset ─────────────────────────────────────────────────────────────
        try:
            result = await _try_reset(env, task_name)
        except Exception as exc:
            print(f"[DEBUG] All reset() attempts failed: {exc}", flush=True)
            log_step(step=1, action="reset_failed", reward=0.0, done=True, error=str(exc)[:100])
            return {"task": task_name, "score": 0.0, "success": False, "steps": 0}

        # ── Episode loop ──────────────────────────────────────────────────────
        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Format observation → LLM prompt
            obs_text = format_observation(result.observation)

            # LLM call
            llm_text = call_llm(client, system, obs_text)
            action   = parse_llm_response(llm_text)

            # Step environment
            step_result, error = await _try_step(env, action, task_name)

            if step_result is not None:
                result = step_result
                reward = float(result.reward or 0.0)
                done   = result.done
            else:
                reward = 0.0
                done   = True

            rewards.append(reward)
            steps_taken = step

            action_summary = (
                f"{action.decision}|"
                f"flags={'+'.join(action.defect_flags) if action.defect_flags else 'NONE'}|"
                f"priority={action.priority}"
            )
            log_step(
                step   = step,
                action = action_summary,
                reward = reward,
                done   = done,
                error  = error,
            )

            if done:
                break

        # ── Final score ───────────────────────────────────────────────────────
        score   = max(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled exception in run_task({task_name}): {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        # [END] always emitted — even on exception
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return {
        "task":    task_name,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Run all 3 tasks sequentially. Never raises."""
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks   = [
        "single_cylinder_triage",
        "batch_inspection",
        "incident_root_cause",
    ]
    results = []

    for i, task in enumerate(tasks):
        if i > 0:
            print(f"[DEBUG] Waiting {RETRY_DELAY_S}s before next task...", flush=True)
            await asyncio.sleep(RETRY_DELAY_S)

        try:
            result = await run_task(task, client)
        except Exception as exc:
            print(f"[DEBUG] run_task({task}) raised: {exc}", flush=True)
            result = {"task": task, "score": 0.0, "success": False, "steps": 0}

        results.append(result)
        print(flush=True)

    # Summary
    print("[SUMMARY]", flush=True)
    total_score = 0.0
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task']:30s} score={r['score']:.3f} "
            f"steps={r['steps']} status={status}",
            flush=True,
        )
        total_score += r["score"]

    avg = total_score / len(results)
    print(f"  {'AVERAGE':30s} score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())