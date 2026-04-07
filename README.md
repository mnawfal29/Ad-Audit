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

Final score (0-1) is weighted:
- **Fraud detection accuracy** (50%) — Correct flags with right fraud type
- **Detection timeliness** (30%) — How early fraud was caught
- **Investigation efficiency** (20%) — Budget usage and false positive avoidance

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
