"""
Publisher engine — per-publisher traffic generation each day.

All randomness is pre-baked into ``day_factors`` and ``noise_factors``
arrays in the case profile, ensuring full determinism.
"""

from __future__ import annotations

from typing import Any, Dict

from .fraud_engine import compute_fraud_intensity


def generate_daily_traffic(
    day: int,
    publisher_cfg: Dict[str, Any],
    budget_allocation: float,
    adaptation_stage: str,
    is_paused: bool,
) -> Dict[str, Any]:
    """Generate one day of traffic for a single publisher.

    Returns a dict with keys:
        impressions, clicks, conversions, spend,
        ctr, cvr,
        legitimate_spend, fraudulent_spend, legitimate_revenue
    """
    if is_paused:
        return _zero_traffic()

    day_idx = day - 1  # 0-indexed into factor arrays
    day_factors = publisher_cfg.get("day_factors", [1.0] * 30)
    noise_factors = publisher_cfg.get("noise_factors", [1.0] * 30)

    day_factor = day_factors[day_idx] if day_idx < len(day_factors) else 1.0
    noise_factor = noise_factors[day_idx] if day_idx < len(noise_factors) else 1.0

    base_rate: float = publisher_cfg["base_traffic_rate"]
    true_ctr: float = publisher_cfg["true_ctr"]
    true_cvr: float = publisher_cfg["true_cvr"]
    cpm_rate: float = publisher_cfg.get("cpm_rate", 2.0)
    conversion_value: float = publisher_cfg.get("conversion_value", 10.0)

    # --- Legitimate traffic ---
    legit_impressions = base_rate * budget_allocation * day_factor * noise_factor
    legit_clicks = legit_impressions * true_ctr * noise_factor
    legit_conversions = legit_clicks * true_cvr * noise_factor
    legit_spend = legit_impressions * cpm_rate / 1000.0
    legit_revenue = legit_conversions * conversion_value

    # --- Fraudulent traffic (only for fraudulent publishers) ---
    fraud_impressions = 0.0
    fraud_clicks = 0.0
    fraud_conversions = 0.0
    fraud_spend = 0.0

    if publisher_cfg.get("is_fraudulent", False):
        fraud_schedule = publisher_cfg.get("fraud_schedule", {})
        if fraud_schedule:
            intensity = compute_fraud_intensity(day, fraud_schedule, adaptation_stage)
            if intensity > 0:
                fake_ctr = publisher_cfg.get("fake_ctr", 0.045)
                fake_cvr = publisher_cfg.get("fake_cvr", 0.001)

                fraud_impressions = legit_impressions * intensity
                fraud_clicks = fraud_impressions * fake_ctr
                fraud_conversions = fraud_clicks * fake_cvr
                fraud_spend = fraud_impressions * cpm_rate / 1000.0

    total_impressions = int(round(legit_impressions + fraud_impressions))
    total_clicks = int(round(legit_clicks + fraud_clicks))
    total_conversions = int(round(legit_conversions + fraud_conversions))
    total_spend = legit_spend + fraud_spend

    ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
    cvr = total_conversions / total_clicks if total_clicks > 0 else 0.0

    return {
        "impressions": total_impressions,
        "clicks": total_clicks,
        "conversions": total_conversions,
        "spend": round(total_spend, 2),
        "ctr": round(ctr, 4),
        "cvr": round(cvr, 4),
        "legitimate_spend": round(legit_spend, 2),
        "fraudulent_spend": round(fraud_spend, 2),
        "legitimate_revenue": round(legit_revenue, 2),
    }


def _zero_traffic() -> Dict[str, Any]:
    return {
        "impressions": 0,
        "clicks": 0,
        "conversions": 0,
        "spend": 0.0,
        "ctr": 0.0,
        "cvr": 0.0,
        "legitimate_spend": 0.0,
        "fraudulent_spend": 0.0,
        "legitimate_revenue": 0.0,
    }
