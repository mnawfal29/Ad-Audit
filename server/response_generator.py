"""Investigation tool response generator.

Returns structured numerical metrics deterministically via SHA256 seeding.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

# ── Legitimate metric ranges ────────────────────────────────────────────────

LEGIT_RANGES: Dict[str, Dict[str, tuple]] = {
    "click_timestamps": {
        "avg_interval_seconds": (30.0, 90.0),
        "interval_std_dev": (15.0, 45.0),
        "pct_clicks_2am_5am": (0.02, 0.08),
        "weekday_weekend_ratio": (1.2, 2.5),
        "pct_sub_second_pairs": (0.0, 0.02),
        "hourly_entropy": (3.0, 3.8),
    },
    "ip_distribution": {
        "unique_ips_per_1000_clicks": (600.0, 950.0),
        "top_ip_pct": (0.5, 3.0),
        "pct_datacenter_ips": (2.0, 8.0),
        "pct_residential_ips": (85.0, 95.0),
        "country_entropy": (1.5, 3.0),
        "pct_ip_subnet_collision": (1.0, 5.0),
    },
    "device_fingerprints": {
        "unique_fps_per_1000_clicks": (700.0, 950.0),
        "top_fp_pct": (0.3, 2.0),
        "pct_headless_browser": (0.0, 0.5),
        "avg_screen_resolutions": (8.0, 25.0),
        "pct_mismatched_timezone_ip": (1.0, 5.0),
        "os_entropy": (1.5, 2.5),
    },
    "referral_urls": {
        "pct_direct_navigation": (15.0, 40.0),
        "pct_referral_domain_mismatch": (1.0, 5.0),
        "unique_referral_domains": (50.0, 200.0),
        "pct_referral_chain_length_gt_2": (1.0, 5.0),
        "referral_domain_entropy": (3.0, 4.5),
    },
    "viewability_scores": {
        "pct_in_viewport_gt_1s": (60.0, 85.0),
        "avg_viewport_dwell_seconds": (3.0, 12.0),
        "pct_zero_pixel_ads": (0.0, 0.5),
        "pct_stacked_ads": (0.0, 1.0),
        "avg_focus_time_seconds": (5.0, 20.0),
        "pct_mouse_nearby": (30.0, 60.0),
    },
    "conversion_quality": {
        "click_to_conversion_seconds_mean": (120.0, 1800.0),
        "conversion_rate": (1.0, 8.0),
        "pct_bounce_after_click": (30.0, 55.0),
        "avg_pages_per_session": (2.5, 6.0),
        "pct_prior_engagement": (20.0, 50.0),
        "pct_last_click_attributed": (40.0, 70.0),
    },
}

# ── Fraud metric ranges ─────────────────────────────────────────────────────
# fraud_type -> adaptation_stage -> tool -> metric -> (lo, hi)
# Only distinctive signals are defined; unlisted combos fall through to LEGIT.

FRAUD_RANGES: Dict[str, Dict[str, Dict[str, Dict[str, tuple]]]] = {
    "bot_traffic": {
        "normal": {
            "click_timestamps": {
                "avg_interval_seconds": (2.0, 5.0),
                "interval_std_dev": (0.5, 2.0),
                "pct_clicks_2am_5am": (0.20, 0.45),
                "pct_sub_second_pairs": (0.10, 0.35),
                "hourly_entropy": (1.0, 2.0),
            },
            "ip_distribution": {
                "unique_ips_per_1000_clicks": (50.0, 200.0),
                "top_ip_pct": (10.0, 35.0),
                "pct_datacenter_ips": (60.0, 90.0),
                "pct_residential_ips": (10.0, 35.0),
                "pct_ip_subnet_collision": (20.0, 50.0),
            },
            "device_fingerprints": {
                "unique_fps_per_1000_clicks": (20.0, 100.0),
                "top_fp_pct": (15.0, 45.0),
                "pct_headless_browser": (30.0, 80.0),
                "pct_mismatched_timezone_ip": (15.0, 40.0),
                "os_entropy": (0.3, 0.8),
            },
            "referral_urls": {
                "pct_direct_navigation": (60.0, 90.0),
                "unique_referral_domains": (3.0, 15.0),
                "referral_domain_entropy": (0.5, 1.5),
            },
            "viewability_scores": {
                "pct_in_viewport_gt_1s": (10.0, 30.0),
                "avg_viewport_dwell_seconds": (0.2, 1.5),
                "avg_focus_time_seconds": (0.1, 1.0),
                "pct_mouse_nearby": (0.0, 5.0),
            },
            "conversion_quality": {
                "conversion_rate": (0.0, 0.1),
                "pct_bounce_after_click": (85.0, 99.0),
                "avg_pages_per_session": (1.0, 1.2),
                "pct_prior_engagement": (0.0, 2.0),
            },
        },
        "cautious": {
            "click_timestamps": {
                "avg_interval_seconds": (8.0, 20.0),
                "interval_std_dev": (3.0, 10.0),
                "pct_clicks_2am_5am": (0.10, 0.25),
                "pct_sub_second_pairs": (0.05, 0.15),
                "hourly_entropy": (2.0, 2.8),
            },
            "ip_distribution": {
                "unique_ips_per_1000_clicks": (200.0, 400.0),
                "top_ip_pct": (5.0, 15.0),
                "pct_datacenter_ips": (30.0, 55.0),
                "pct_residential_ips": (40.0, 65.0),
                "pct_ip_subnet_collision": (10.0, 25.0),
            },
            "device_fingerprints": {
                "unique_fps_per_1000_clicks": (200.0, 450.0),
                "top_fp_pct": (5.0, 15.0),
                "pct_headless_browser": (10.0, 30.0),
                "pct_mismatched_timezone_ip": (8.0, 20.0),
                "os_entropy": (0.8, 1.3),
            },
            "referral_urls": {
                "pct_direct_navigation": (40.0, 65.0),
                "unique_referral_domains": (15.0, 40.0),
                "referral_domain_entropy": (1.5, 2.5),
            },
            "viewability_scores": {
                "pct_in_viewport_gt_1s": (25.0, 45.0),
                "avg_viewport_dwell_seconds": (1.0, 3.0),
                "avg_focus_time_seconds": (1.0, 3.0),
                "pct_mouse_nearby": (5.0, 15.0),
            },
            "conversion_quality": {
                "conversion_rate": (0.1, 0.5),
                "pct_bounce_after_click": (70.0, 85.0),
                "avg_pages_per_session": (1.2, 1.8),
                "pct_prior_engagement": (2.0, 8.0),
            },
        },
        "covering_tracks": {
            "click_timestamps": {
                "avg_interval_seconds": (18.0, 35.0),
                "interval_std_dev": (8.0, 18.0),
                "pct_clicks_2am_5am": (0.06, 0.12),
                "pct_sub_second_pairs": (0.02, 0.06),
                "hourly_entropy": (2.5, 3.2),
            },
            "ip_distribution": {
                "unique_ips_per_1000_clicks": (400.0, 600.0),
                "top_ip_pct": (3.0, 8.0),
                "pct_datacenter_ips": (12.0, 25.0),
                "pct_residential_ips": (65.0, 82.0),
                "pct_ip_subnet_collision": (5.0, 12.0),
            },
            "device_fingerprints": {
                "unique_fps_per_1000_clicks": (450.0, 650.0),
                "top_fp_pct": (2.0, 6.0),
                "pct_headless_browser": (2.0, 8.0),
                "pct_mismatched_timezone_ip": (4.0, 10.0),
                "os_entropy": (1.2, 1.8),
            },
            "conversion_quality": {
                "conversion_rate": (0.5, 1.5),
                "pct_bounce_after_click": (55.0, 70.0),
                "avg_pages_per_session": (1.5, 2.5),
            },
        },
        "dark": {},  # no fraud signals — looks legit
    },
    "domain_spoofing": {
        "normal": {
            "referral_urls": {
                "pct_referral_domain_mismatch": (40.0, 75.0),
                "pct_referral_chain_length_gt_2": (15.0, 35.0),
                "referral_domain_entropy": (0.8, 1.8),
            },
            "viewability_scores": {
                "pct_zero_pixel_ads": (15.0, 45.0),
                "pct_stacked_ads": (10.0, 30.0),
                "pct_in_viewport_gt_1s": (15.0, 35.0),
                "avg_viewport_dwell_seconds": (0.5, 2.0),
            },
            "ip_distribution": {
                "pct_datacenter_ips": (25.0, 50.0),
                "pct_residential_ips": (45.0, 70.0),
            },
            "device_fingerprints": {
                "pct_headless_browser": (5.0, 20.0),
            },
            "click_timestamps": {
                "avg_interval_seconds": (15.0, 35.0),
                "pct_clicks_2am_5am": (0.10, 0.20),
            },
            "conversion_quality": {
                "pct_bounce_after_click": (65.0, 85.0),
                "avg_pages_per_session": (1.2, 2.0),
            },
        },
        "cautious": {
            "referral_urls": {
                "pct_referral_domain_mismatch": (20.0, 40.0),
                "pct_referral_chain_length_gt_2": (8.0, 18.0),
                "referral_domain_entropy": (1.8, 2.8),
            },
            "viewability_scores": {
                "pct_zero_pixel_ads": (5.0, 15.0),
                "pct_stacked_ads": (3.0, 10.0),
                "pct_in_viewport_gt_1s": (35.0, 50.0),
                "avg_viewport_dwell_seconds": (2.0, 4.0),
            },
        },
        "covering_tracks": {
            "referral_urls": {
                "pct_referral_domain_mismatch": (8.0, 18.0),
                "pct_referral_chain_length_gt_2": (4.0, 8.0),
            },
            "viewability_scores": {
                "pct_zero_pixel_ads": (1.0, 5.0),
                "pct_stacked_ads": (1.0, 3.0),
                "pct_in_viewport_gt_1s": (45.0, 60.0),
            },
        },
        "dark": {},
    },
    "click_injection": {
        "normal": {
            "conversion_quality": {
                "click_to_conversion_seconds_mean": (2.0, 15.0),
                "conversion_rate": (15.0, 50.0),
                "pct_last_click_attributed": (85.0, 99.0),
                "pct_bounce_after_click": (10.0, 25.0),
                "avg_pages_per_session": (1.0, 1.5),
            },
            "click_timestamps": {
                "avg_interval_seconds": (5.0, 15.0),
                "pct_sub_second_pairs": (0.15, 0.40),
                "hourly_entropy": (1.5, 2.5),
            },
            "device_fingerprints": {
                "pct_headless_browser": (5.0, 25.0),
                "pct_mismatched_timezone_ip": (10.0, 25.0),
            },
            "ip_distribution": {
                "pct_datacenter_ips": (15.0, 35.0),
            },
        },
        "cautious": {
            "conversion_quality": {
                "click_to_conversion_seconds_mean": (15.0, 60.0),
                "conversion_rate": (8.0, 20.0),
                "pct_last_click_attributed": (70.0, 85.0),
                "pct_bounce_after_click": (25.0, 40.0),
            },
            "click_timestamps": {
                "avg_interval_seconds": (12.0, 25.0),
                "pct_sub_second_pairs": (0.05, 0.15),
                "hourly_entropy": (2.2, 3.0),
            },
            "device_fingerprints": {
                "pct_headless_browser": (2.0, 8.0),
                "pct_mismatched_timezone_ip": (5.0, 12.0),
            },
        },
        "covering_tracks": {
            "conversion_quality": {
                "click_to_conversion_seconds_mean": (50.0, 120.0),
                "conversion_rate": (5.0, 10.0),
                "pct_last_click_attributed": (60.0, 72.0),
            },
            "click_timestamps": {
                "avg_interval_seconds": (20.0, 35.0),
                "pct_sub_second_pairs": (0.02, 0.06),
            },
        },
        "dark": {},
    },
}


# ── Seeded value generator ──────────────────────────────────────────────────

def _seeded_value(seed_str: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    t = (h % 10000) / 10000.0
    return round(lo + t * (hi - lo), 4)


# ── Public API ──────────────────────────────────────────────────────────────

def generate_investigation_metrics(
    case_id: str,
    publisher_id: str,
    publisher_cfg: Dict[str, Any],
    tool_name: str,
    adaptation_stage: str,
) -> Dict[str, Any]:
    """Return structured numerical metrics for an investigation tool."""
    is_fraudulent = publisher_cfg.get("is_fraudulent", False)
    fraud_type = publisher_cfg.get("fraud_type")

    legit = LEGIT_RANGES.get(tool_name, {})
    if not legit:
        return {"error": f"Unknown tool: {tool_name}"}

    # Determine which ranges to use
    fraud_tool_ranges: Dict[str, tuple] = {}
    if is_fraudulent and fraud_type and adaptation_stage != "dark":
        type_ranges = FRAUD_RANGES.get(fraud_type, {})
        stage_ranges = type_ranges.get(adaptation_stage, {})
        fraud_tool_ranges = stage_ranges.get(tool_name, {})

    metrics: Dict[str, Any] = {}
    for metric_name, legit_range in legit.items():
        seed = f"{case_id}:{publisher_id}:{tool_name}:{metric_name}"
        if metric_name in fraud_tool_ranges:
            lo, hi = fraud_tool_ranges[metric_name]
        else:
            lo, hi = legit_range
        metrics[metric_name] = _seeded_value(seed, lo, hi)

    return {
        "tool": tool_name,
        "publisher_id": publisher_id,
        "metrics": metrics,
    }


def generate_trend_summary() -> str:
    """Placeholder for trend summary (currently muted)."""
    return ""


def generate_alerts() -> list:
    """Placeholder for alerts (currently muted)."""
    return []
