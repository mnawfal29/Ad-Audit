"""
Fraud engine — suspicion tracking and fraud intensity for adaptive publishers.
"""

from __future__ import annotations

from typing import Any, Dict


# ── Suspicion tracking ────────────────────────────────────────────────────

def update_suspicion(
    current_level: float,
    tool: str,
    reactivity: float,
) -> float:
    """Increase suspicion after an investigation tool is used on a fraudster.

    Each tool adds a fixed bump scaled by the publisher's reactivity.
    """
    bump = {
        "click_timestamps": 0.15,
        "ip_distribution": 0.12,
        "device_fingerprints": 0.10,
        "referral_urls": 0.10,
        "viewability_scores": 0.08,
        "conversion_quality": 0.10,
    }.get(tool, 0.10)
    return min(1.0, current_level + bump * reactivity)


def decay_suspicion(level: float, rate: float = 0.05) -> float:
    """Decay suspicion each day a fraudster is NOT investigated."""
    return max(0.0, level - rate)


def get_adaptation_stage(suspicion_level: float) -> str:
    """Map suspicion level to an adaptation stage for response generation."""
    if suspicion_level >= 0.8:
        return "dark"
    if suspicion_level >= 0.5:
        return "covering_tracks"
    if suspicion_level >= 0.25:
        return "cautious"
    return "normal"


# ── Fraud intensity ───────────────────────────────────────────────────────

def compute_fraud_intensity(
    day: int,
    fraud_schedule: Dict[str, Any],
    adaptation_stage: str,
) -> float:
    """Compute how aggressively a publisher is committing fraud on a given day.

    ``fraud_schedule`` comes from the case profile and has:
        start_day: int — first day fraud begins
        ramp_days: int — days to ramp from 0 → peak
        peak_intensity: float — maximum multiplier on legitimate traffic
    """
    start = fraud_schedule.get("start_day", 1)
    ramp = fraud_schedule.get("ramp_days", 3)
    peak = fraud_schedule.get("peak_intensity", 1.0)

    if day < start:
        return 0.0

    # Ramp up
    days_active = day - start
    if ramp > 0 and days_active < ramp:
        base = peak * (days_active / ramp)
    else:
        base = peak

    # Adaptation dampening — fraudster backs off when suspicion rises
    stage_mult = {
        "normal": 1.0,
        "cautious": 0.7,
        "covering_tracks": 0.4,
        "dark": 0.05,
    }.get(adaptation_stage, 1.0)

    return base * stage_mult
