"""
Inference Script for AdAudit
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from server.Ad_Audit_environment import AdAuditEnv, EPISODE_DAYS
from models import AdAuditAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

_VALID_TASKS = {"easy", "medium", "hard"}
_task_env = os.getenv("ADAUDIT_TASK", "").strip().lower()
TASK_NAME = _task_env if _task_env in _VALID_TASKS else "medium"
BENCHMARK = os.getenv("ADAUDIT_BENCHMARK", "adaudit")
TEMPERATURE = 0.0
MAX_TOKENS = 2048
HISTORY_WINDOW = 5
BASELINE_DAYS = 3
SUCCESS_SCORE_THRESHOLD = 0.4

# Rule-based investigation tools per fraud type
TOOLS_FOR = {
    "bot_traffic": ["click_timestamps", "ip_distribution"],
    "click_injection": ["conversion_quality", "click_timestamps"],
    "domain_spoofing": ["referral_urls", "viewability_scores"],
}

SYSTEM_PROMPT = """\
You are an ad fraud analyst monitoring a 14-day campaign with multiple publishers.
Each day you see metrics (impressions, clicks, conversions, CTR, CVR) and take ONE action.

## ACTIONS

Use publisher IDs (pub_001, pub_002, etc.), never publisher names.

1. Monitor — observe metrics, take no action:
   {"action_type": "monitor"}
2. Investigate — run one tool on one publisher (costs 1 budget):
   {"action_type": "investigate_publisher", "publisher_id": "pub_001", "tool": "click_timestamps"}
3. Flag fraud — flag a publisher as fraudulent (irreversible):
   {"action_type": "flag_fraud", "publisher_id": "pub_001", "fraud_type": "bot_traffic", "evidence": ["click_timestamps", "ip_distribution"]}
4. Submit report — end the episode:
   {"action_type": "submit_report", "summary": "..."}

Valid tools: click_timestamps, ip_distribution, device_fingerprints, \
referral_urls, viewability_scores, conversion_quality
Valid fraud types: bot_traffic, domain_spoofing, click_injection

## STRATEGY
1. Monitor days 1-3 to establish baselines. Then compare each publisher's CTR/CVR against its own baseline. Only publishers with BIG shifts (CTR doubled, CVR crashed, impressions tripled) are suspicious. Stable metrics = clean.
2. Investigate suspicious publishers with 2 tools before flagging. bot_traffic = CTR spikes + CVR near zero. click_injection = CVR abnormally high (5%+). domain_spoofing = impressions surge + CVR drops.
3. False positives are heavily penalized. When in doubt, monitor. 
4. Do not submit report until you are confident you have found and flagged ALL fraudsters.

Respond with ONLY a JSON action in ```json``` markers.
"""


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Observation formatting (for LLM)
# ---------------------------------------------------------------------------

def format_observation(obs_dict: Dict[str, Any], action_log: List[str] = None) -> str:
    parts: List[str] = []
    parts.append(f"=== DAY {obs_dict['day']} of {obs_dict.get('campaign_day_total', 14)} ===\n")

    if action_log:
        parts.append("Past actions: " + " | ".join(action_log))
        parts.append("")

    if obs_dict.get("daily_metrics"):
        parts.append("Metrics:")
        parts.append(f"{'ID':<10} {'Publisher':<22} {'Impressions':>12} {'Clicks':>8} {'Conversions':>12} {'Spend ($)':>10} {'CTR':>7} {'CVR':>7}")
        for m in obs_dict["daily_metrics"]:
            parts.append(
                f"{m['publisher_id']:<10} {m['name']:<22} {m['impressions']:>12,} {m['clicks']:>8,} "
                f"{m['conversions']:>12,} {m['spend']:>10,.2f} {m['ctr']:>6.2%} {m['cvr']:>6.2%}"
            )
        parts.append("")

    if obs_dict.get("investigation_results"):
        inv = obs_dict["investigation_results"]
        if isinstance(inv, dict):
            if "error" in inv:
                parts.append(f"Investigation ERROR: {inv['error']}")
            else:
                parts.append(f"Investigation ({inv.get('tool', '?')}) for {inv.get('publisher_id', '?')}:")
                for k, v in inv.items():
                    if k not in ("tool", "publisher_id"):
                        parts.append(f"  {k}: {v}")
        parts.append("")

    pub_status = obs_dict.get("publisher_status", {})
    if pub_status:
        flagged_pubs = [pid for pid, s in pub_status.items() if s == "flagged"]
        active = [pid for pid, s in pub_status.items() if s == "active"]
        if flagged_pubs:
            parts.append(f"FLAGGED: {', '.join(flagged_pubs)}")
        parts.append(f"Active publishers: {', '.join(active)}")

    b = obs_dict.get("budget_status", {})
    if b:
        parts.append(
            f"Budget: ${b.get('remaining',0):,.0f} remaining | "
            f"Investigations left: {b.get('investigation_budget_remaining', 0)}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing (for LLM)
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"action_type": "monitor"}


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    obs_text: str,
    history: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Try to get an action from the LLM. Returns None on failure."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content or ""
        return parse_action(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def score_publishers(metrics_history: Dict[str, List[Dict]]) -> List[Tuple[str, float, str]]:
    """Score each publisher's suspicion based on accumulated metrics."""
    results = []
    for pid, hist in metrics_history.items():
        if len(hist) < 2:
            continue
        recent = hist[-3:]
        avg_ctr = sum(h["ctr"] for h in recent) / len(recent)
        avg_cvr = sum(h["cvr"] for h in recent) / len(recent)
        baseline = hist[:2]
        base_ctr = sum(h["ctr"] for h in baseline) / len(baseline)
        base_cvr = sum(h["cvr"] for h in baseline) / len(baseline)

        score = 0.0
        fraud_type = "bot_traffic"

        if base_ctr > 0:
            ctr_rise = avg_ctr / base_ctr
            if ctr_rise > 1.5:
                score += 0.4
            if ctr_rise > 2.0:
                score += 0.3
        if base_cvr > 0:
            cvr_drop = avg_cvr / base_cvr
            if cvr_drop < 0.5:
                score += 0.3
            if cvr_drop < 0.2:
                score += 0.3
        if avg_cvr > 0.04:
            score += 0.5
            fraud_type = "click_injection"
        if base_ctr > 0 and base_cvr > 0:
            ctr_rise = avg_ctr / base_ctr
            cvr_drop = avg_cvr / base_cvr
            if 1.2 < ctr_rise < 2.0 and 0.2 < cvr_drop < 0.6:
                if fraud_type == "bot_traffic" and score < 0.5:
                    fraud_type = "domain_spoofing"

        if score > 0.2:
            results.append((pid, score, fraud_type))

    results.sort(key=lambda x: -x[1])
    return results


def get_rule_action(
    obs_dict: Dict[str, Any],
    metrics_history: Dict[str, List[Dict]],
    investigated: Dict[str, List[str]],
    flagged: set,
) -> Dict[str, Any]:
    """Deterministic rule-based action selection."""
    day = obs_dict["day"]
    budget_left = obs_dict.get("budget_status", {}).get("investigation_budget_remaining", 0)

    if day <= BASELINE_DAYS:
        return {"action_type": "monitor"}

    suspects = score_publishers(metrics_history)
    suspects = [(pid, sc, ft) for pid, sc, ft in suspects if pid not in flagged]

    for pid, _, ft in suspects:
        tools_done = investigated.get(pid, [])
        if len(tools_done) >= 2:
            flagged.add(pid)
            return {
                "action_type": "flag_fraud",
                "publisher_id": pid,
                "fraud_type": ft,
                "evidence": tools_done,
            }

    if budget_left > 0 and suspects:
        for pid, _, ft in suspects:
            if pid in flagged:
                continue
            tools_done = investigated.get(pid, [])
            tools_to_try = TOOLS_FOR.get(ft, TOOLS_FOR["bot_traffic"])
            for tool in tools_to_try:
                if tool not in tools_done:
                    investigated.setdefault(pid, []).append(tool)
                    return {
                        "action_type": "investigate_publisher",
                        "publisher_id": pid,
                        "tool": tool,
                    }

    return {"action_type": "monitor"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Try to init LLM client; fall back to rule-based if it fails
    llm_client: Optional[OpenAI] = None
    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        llm_client.models.list()
    except Exception:
        llm_client = None

    use_rules = llm_client is None

    env = AdAuditEnv()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # LLM state
    history: List[Dict[str, str]] = []
    action_log: List[str] = []

    # Rule-based state
    metrics_history: Dict[str, List[Dict]] = {}
    investigated: Dict[str, List[str]] = {}
    flagged: set = set()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME if not use_rules else "rule-based")

    try:
        obs = env.reset(episode_id=TASK_NAME)
        obs_dict = obs.model_dump()

        while not obs_dict.get("done", False) and steps_taken < EPISODE_DAYS:
            # Track metrics for rule-based fallback
            for m in obs_dict.get("daily_metrics", []):
                metrics_history.setdefault(m["publisher_id"], []).append(m)

            action = None

            if not use_rules:
                obs_text = format_observation(obs_dict, action_log)
                action = get_llm_action(llm_client, obs_text, history[-HISTORY_WINDOW * 2:])
                if action is None:
                    use_rules = True

            if action is None:
                action = get_rule_action(obs_dict, metrics_history, investigated, flagged)

            # Validate action
            try:
                action_obj = AdAuditAction(**action)
            except Exception:
                action_obj = AdAuditAction(action_type="invalid")

            # Build action log entry
            log_entry = f"D{obs_dict['day']}:{action_obj.action_type}"
            if action_obj.publisher_id:
                log_entry += f"({action_obj.publisher_id}"
                if action_obj.tool:
                    log_entry += f",{action_obj.tool}"
                if action_obj.fraud_type:
                    log_entry += f",{action_obj.fraud_type}"
                log_entry += ")"
            action_log.append(log_entry)

            # Update LLM history
            if not use_rules:
                history.append({"role": "user", "content": obs_text})
                history.append({"role": "assistant", "content": json.dumps(action)})

            # Step environment
            obs = env.step(action_obj)
            obs_dict = obs.model_dump()
            steps_taken += 1

            reward = obs_dict.get("reward", 0.0)
            done = obs_dict.get("done", False)
            error = None
            rewards.append(reward)

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

        # Final grading
        state = env.state
        grader = state.grader_inputs
        score = grader.get("final_score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
