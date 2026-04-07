"""Episode-end grader [0, 1].

Three components:
  1. Fraud detection accuracy  (weight 0.50)
  2. Detection timeliness      (weight 0.30)
  3. Investigation efficiency   (weight 0.20)
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade_episode(
    state_dict: Dict[str, Any],
    case_dict: Dict[str, Any],
) -> Dict:
    """Return grading breakdown and final score in [0, 1].

    Args:
        state_dict: Result of _build_grader_state() — enriched AdAuditState dict.
        case_dict: The raw case JSON (with publisher configs).
    """
    publishers = state_dict.get("publishers", [])
    flags_submitted = state_dict.get("flags_submitted", [])
    investigation_budget_used = state_dict.get("investigation_budget_used", 0)
    investigation_budget_total = state_dict.get("investigation_budget_total", 0)

    # Derive fraud info from the enriched publisher entries
    fraudulent_publishers: List[str] = []
    fraud_types: Dict[str, str] = {}
    publisher_internals: Dict[str, Dict] = {}
    tools_used_per_publisher: Dict[str, List[str]] = {}

    case_publishers = case_dict.get("publishers", {})

    for pub in publishers:
        pid = pub["publisher_id"]
        tools_used_per_publisher[pid] = pub.get("tools_used", [])

        if pub.get("is_fraudulent"):
            fraudulent_publishers.append(pid)
            fraud_types[pid] = pub.get("fraud_type", "")

            # Get fraud_start_day from case config's fraud_schedule
            cfg = case_publishers.get(pid, {})
            fraud_schedule = cfg.get("fraud_schedule", {})
            publisher_internals[pid] = {
                "fraud_start_day": fraud_schedule.get("start_day", 1),
            }

    num_fraudulent = len(fraudulent_publishers)
    if num_fraudulent == 0:
        return {
            "accuracy": 1.0,
            "timeliness": 1.0,
            "efficiency": 1.0,
            "final_score": 1.0,
            "num_fraudulent": 0,
            "num_flagged_correct": 0,
            "num_false_positives": 0,
        }

    # Build lookup of flags by publisher_id
    flag_lookup: Dict[str, Dict] = {}
    for f in flags_submitted:
        flag_lookup[f["publisher_id"]] = f

    # Count false positives
    flagged_ids = {f["publisher_id"] for f in flags_submitted}
    false_positives = [pid for pid in flagged_ids if pid not in fraudulent_publishers]
    num_false_positives = len(false_positives)

    # ── 1. Fraud Detection Accuracy ─────────────────────────────────────
    accuracy = 0.0
    for pid in fraudulent_publishers:
        if pid in flag_lookup:
            if flag_lookup[pid].get("type_correct"):
                accuracy += 1.0 / num_fraudulent
            else:
                accuracy += 0.5 / num_fraudulent
    # Penalty for false positives
    accuracy -= num_false_positives * (0.5 / num_fraudulent)
    accuracy = max(0.0, min(1.0, accuracy))

    # ── 2. Detection Timeliness ─────────────────────────────────────────
    timeliness_scores = []
    for pid in fraudulent_publishers:
        internal = publisher_internals.get(pid, {})
        fraud_start = internal.get("fraud_start_day", 1)
        if pid in flag_lookup:
            day_flagged = flag_lookup[pid].get("day", 14)
            denom = 14 - fraud_start
            if denom <= 0:
                t = 1.0
            else:
                t = 1.0 - (day_flagged - fraud_start) / denom
            timeliness_scores.append(max(0.0, min(1.0, t)))
        else:
            timeliness_scores.append(0.0)
    timeliness = sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.0

    # ── 3. Investigation Efficiency ─────────────────────────────────────
    total = investigation_budget_used
    useful = 0
    for pid in fraudulent_publishers:
        useful += len(tools_used_per_publisher.get(pid, []))

    if total > 0:
        information_value = useful / total
    else:
        information_value = 0.0

    if investigation_budget_total > 0:
        budget_efficiency = 1.0 - (total / investigation_budget_total)
    else:
        budget_efficiency = 1.0

    fp_penalty = num_false_positives * 0.2
    efficiency = 0.5 * information_value + 0.3 * budget_efficiency - fp_penalty
    efficiency = max(0.0, min(1.0, efficiency))

    # ── Final Score ─────────────────────────────────────────────────────
    final = min(1.0, 0.50 * accuracy + 0.30 * timeliness + 0.20 * efficiency)

    return {
        "accuracy": round(accuracy, 4),
        "timeliness": round(timeliness, 4),
        "efficiency": round(efficiency, 4),
        "final_score": round(final, 4),
        "num_fraudulent": num_fraudulent,
        "num_flagged_correct": sum(1 for pid in fraudulent_publishers if pid in flag_lookup),
        "num_false_positives": num_false_positives,
    }
