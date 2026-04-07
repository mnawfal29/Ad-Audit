"""Per-step reward calculator [0, 1]."""


def compute_step_reward(
    action_type: str,
    daily_fraud_spend: float,
    total_budget: float,
    publisher_cfg: dict | None = None,
    flag_correct: bool | None = None,
    flag_type_correct: bool | None = None,
    day: int = 1,
    episode_days: int = 14,
) -> float:
    """Return a reward in [0.0, 1.0] for a single step.

    Centered at 0.5 (neutral). Rewards scale with timing and precision.
    """
    if action_type == "monitor":
        if daily_fraud_spend > 0:
            # Active unflagged fraud — penalty grows with time (urgency)
            progress = day / episode_days  # 0.07 on day 1, 1.0 on day 14
            penalty = 0.10 + 0.20 * progress  # 0.10 early, up to 0.30 late
            return max(0.05, 0.50 - penalty)
        return 0.50

    if action_type == "investigate_publisher":
        if publisher_cfg is not None and publisher_cfg.get("is_fraudulent"):
            # Investigating a real fraudster — reward scales with how early
            early_bonus = max(0.0, (episode_days - day) / episode_days) * 0.10
            return min(1.0, 0.55 + early_bonus)
        # Investigating a clean publisher — wastes budget
        return 0.35

    if action_type == "flag_fraud":
        if flag_correct is True and flag_type_correct is True:
            # Perfect flag — bonus for catching it early
            early_bonus = max(0.0, (episode_days - day) / episode_days) * 0.05
            return min(1.0, 0.95 + early_bonus)
        if flag_correct is True:
            # Right publisher, wrong type
            return 0.70
        # False positive — heavy penalty
        return 0.05

    if action_type == "submit_report":
        return 0.50

    # invalid / malformed
    return 0.05
