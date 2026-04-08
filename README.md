---
title: Ad Audit Environment
emoji: 🕵️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ad Audit Environment

An RL environment for **detecting advertising fraud** in a simulated 14-day ad campaign. Agents monitor publisher traffic metrics, investigate suspicious patterns, and flag fraudulent publishers while avoiding false positives.

## The Challenge

You manage a digital ad campaign with multiple publishers. Some are legitimate, some are committing fraud. Each day you see traffic metrics and must decide: monitor, investigate, or flag.

**Fraud types:**
- **Bot Traffic** — CTR spikes dramatically, CVR drops near zero (bots click but never convert)
- **Click Injection** — CVR becomes abnormally high (fake conversions injected)
- **Domain Spoofing** — Impressions surge while CVR drops (fake ad inventory)

**The catch:** False positives are heavily penalized, investigations cost budget, and fraudsters adapt when investigated.

## Quick Start

```python
import asyncio
from Ad_Audit import AdAuditAction, AdAuditEnv

async def main():
    env = await AdAuditEnv.from_docker_image("adaudit-env:latest")
    try:
        result = await env.reset(episode_id="medium")
        obs = result.observation
        print(f"Day {obs.day}: {len(obs.daily_metrics)} publishers")

        # Monitor day 1
        result = await env.step(AdAuditAction(action_type="monitor"))

        # Investigate a suspicious publisher
        result = await env.step(AdAuditAction(
            action_type="investigate_publisher",
            publisher_id="pub_003",
            tool="click_timestamps"
        ))

        # Flag fraud with evidence
        result = await env.step(AdAuditAction(
            action_type="flag_fraud",
            publisher_id="pub_003",
            fraud_type="bot_traffic",
            evidence=["click_timestamps", "ip_distribution"]
        ))
        print(f"Reward: {result.reward}")
    finally:
        await env.close()

asyncio.run(main())
```

## Actions

| Action | Description | Cost |
|--------|-------------|------|
| `monitor` | Observe metrics, take no action | Free |
| `investigate_publisher` | Run a tool on one publisher | 1 investigation budget |
| `flag_fraud` | Flag publisher as fraudulent (irreversible) | Free but false positives penalized |
| `submit_report` | End the episode early | Free |

**Investigation tools:** click_timestamps, ip_distribution, device_fingerprints, referral_urls, viewability_scores, conversion_quality

**Fraud types:** bot_traffic, domain_spoofing, click_injection

### Action input format

Each action is a JSON object with the fields below. Only include fields relevant to the chosen `action_type`.

| Field | Type | Used by | Example |
|-------|------|---------|---------|
| `action_type` | string (required) | all | `"monitor"` |
| `publisher_id` | string | investigate, flag | `"pub_001"` |
| `tool` | string | investigate | `"click_timestamps"` |
| `fraud_type` | string | flag | `"bot_traffic"` |
| `evidence` | list of strings | flag | `["click_timestamps", "ip_distribution"]` |
| `summary` | string | submit_report | `"Publisher pub_002 is running bot traffic"` |

**Examples:**

```json
{"action_type": "monitor"}
```

```json
{"action_type": "investigate_publisher", "publisher_id": "pub_001", "tool": "click_timestamps"}
```

```json
{
  "action_type": "flag_fraud",
  "publisher_id": "pub_002",
  "fraud_type": "bot_traffic",
  "evidence": ["click_timestamps", "ip_distribution"]
}
```

```json
{"action_type": "submit_report", "summary": "Flagged pub_002 for bot traffic based on timestamp clustering and IP concentration."}
```

> **Web UI note:** In the web interface, fill in only the fields for your chosen action and leave the rest blank. For the `evidence` field, enter tool names separated by commas (e.g. `click_timestamps, ip_distribution`).

## Observation

Each step returns:
- **daily_metrics** — Per-publisher: impressions, clicks, conversions, spend, CTR, CVR
- **investigation_results** — Tool output (if investigated)
- **publisher_status** — Active or flagged
- **budget_status** — Campaign spend and remaining investigation budget

## Tasks

| Task | Publishers | Fraudsters | Investigation Budget | Difficulty |
|------|-----------|------------|---------------------|------------|
| `easy` | 2 | 1 (bot_traffic) | 10 | Obvious signals |
| `medium` | 4 | 2 (bot_traffic + click_injection) | 10 | Mixed fraud types |
| `hard` | 4 | 2 (domain_spoofing + bot_traffic) | 6 | Subtle signals, tight budget |

## Scoring

### Step Reward

Every action returns an immediate reward in **[0, 1]**, centered at 0.5 (neutral).

| Action | Condition | Reward |
|--------|-----------|--------|
| `monitor` | No active fraud | 0.50 |
| `monitor` | Active unflagged fraud | 0.40 → 0.20 (penalty grows day over day) |
| `investigate_publisher` | Publisher is fraudulent | 0.55 → 0.65 (bonus for investigating early) |
| `investigate_publisher` | Publisher is clean | 0.35 (wastes budget) |
| `flag_fraud` | Correct publisher + correct fraud type | 0.95 → 1.00 (bonus for early flag) |
| `flag_fraud` | Correct publisher, wrong fraud type | 0.70 |
| `flag_fraud` | False positive | 0.05 |
| `submit_report` | Any | 0.50 |
| Invalid / malformed action | — | 0.05 |

The monitor penalty formula: `0.50 - (0.10 + 0.20 × day/14)`, floored at 0.05. On day 1 the penalty is ~0.10; by day 14 it reaches ~0.30, reflecting increasing urgency as fraud compounds.

### Final Score

Computed at episode end, combining three weighted components into a score in **[0, 1]**:

```
final_score = 0.50 × accuracy + 0.30 × timeliness + 0.20 × efficiency
```

#### 1. Fraud Detection Accuracy (50%)

Measures whether fraudulent publishers were correctly identified with the right fraud type.

- **+1.0 / N** per fraudster flagged with the correct fraud type
- **+0.5 / N** per fraudster flagged with the wrong fraud type
- **−0.5 / N** per false positive (clean publisher flagged as fraudulent)

Clamped to [0, 1].

#### 2. Detection Timeliness (30%)

Measures how quickly each fraudster was caught after fraud began.

```
timeliness = 1.0 − (day_flagged − fraud_start_day) / (14 − fraud_start_day)
```

- Flagging immediately when fraud starts → 1.0
- Flagging on the final day → 0.0
- Unflagged fraudster → 0.0
- Averaged across all fraudsters.

#### 3. Investigation Efficiency (20%)

Measures whether investigations were targeted at real fraudsters without wasting budget.

```
efficiency = 0.5 × (useful_investigations / total_investigations)
           + 0.3 × (1 − budget_used / budget_total)
           − 0.2 × num_false_positives
```

- **Information value** — fraction of investigations spent on fraudulent publishers
- **Budget efficiency** — fraction of budget left unused
- **False positive penalty** — −0.2 per clean publisher incorrectly flagged

Clamped to [0, 1].

## Deployment

```bash
# Build Docker image
docker build -t adaudit-env .

# Run locally
docker run -p 8000:8000 adaudit-env

# Or without Docker
ENABLE_WEB_INTERFACE=true python -m server.app
```

**Endpoints:**
- `/web` — Interactive Gradio UI
- `/docs` — API documentation
- `/health` — Health check
- `/ws` — WebSocket for persistent sessions

## Project Structure

```
Ad_Audit/
├── inference.py           # LLM agent + rule-based fallback
├── models.py              # Action / Observation / State models
├── client.py              # WebSocket client (AdAuditEnv)
├── cases/                 # Task definitions (easy/medium/hard)
└── server/
    ├── app.py             # FastAPI server
    ├── Ad_Audit_environment.py  # Core environment logic
    ├── fraud_engine.py    # Suspicion tracking & fraud intensity
    ├── publisher_engine.py # Traffic generation
    ├── response_generator.py # Investigation tool responses
    ├── step_reward.py     # Per-step reward calculator
    └── grader.py          # Episode-end scoring
```
