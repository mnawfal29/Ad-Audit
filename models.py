"""
Pydantic models for the AdAudit environment.

Defines Action, Observation, and State types that conform to the OpenEnv spec.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AdAuditAction(Action):
    """Single discrete action the agent takes each day."""

    action_type: Literal[
        "monitor",
        "investigate_publisher",
        "flag_fraud",
        "submit_report",
        "invalid",
    ] = Field(..., description="The type of action to take")

    publisher_id: Optional[str] = Field(
        default=None, description="Target publisher for investigate/flag actions"
    )

    # investigate_publisher
    tool: Optional[Literal[
        "click_timestamps",
        "ip_distribution",
        "device_fingerprints",
        "referral_urls",
        "viewability_scores",
        "conversion_quality",
    ]] = Field(default=None, description="Investigation tool to use")

    # flag_fraud
    fraud_type: Optional[Literal[
        "bot_traffic",
        "domain_spoofing",
        "click_injection",
    ]] = Field(default=None, description="Fraud type to flag")
    evidence: Optional[List[str]] = Field(
        default=None,
        description="Evidence tool names, comma-separated (e.g. click_timestamps, ip_distribution)",
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v):
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Handle bare string like "click_timestamps"
            stripped = v.strip("[] ")
            return [s.strip().strip("'\"") for s in stripped.split(",") if s.strip()]
        return v

    # submit_report
    summary: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

class DailyPublisherMetrics(BaseModel):
    """Traffic metrics for one publisher on one day."""

    publisher_id: str
    name: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    ctr: float
    cvr: float


class BudgetStatus(BaseModel):
    """Campaign and investigation budget snapshot."""

    total_campaign_budget: float
    spent_so_far: float
    remaining: float
    investigation_budget_remaining: int
    daily_spend_rate: float


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class AdAuditObservation(Observation):
    """What the agent sees after each step.

    Inherits ``done``, ``reward``, and ``metadata`` from the OpenEnv
    ``Observation`` base class.  ``reward`` carries the daily P&L.
    """

    day: int = Field(..., description="Current campaign day (1-30)")
    campaign_day_total: int = Field(default=14)

    daily_metrics: List[DailyPublisherMetrics] = Field(default_factory=list)
    cumulative_metrics: List[DailyPublisherMetrics] = Field(default_factory=list)

    trend_data: str = Field(default="", description="Trend summary")
    investigation_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured metrics from investigation tool"
    )
    alerts: List[str] = Field(default_factory=list)

    budget_status: Optional[BudgetStatus] = None
    publisher_status: Dict[str, str] = Field(
        default_factory=dict,
        description="publisher_id -> active|flagged",
    )

    cumulative_reward: float = Field(default=0.0)
    done_reason: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# State (hidden — used for grading / debugging)
# ---------------------------------------------------------------------------

class PublisherState(BaseModel):
    """Public publisher state (visible via /state)."""

    publisher_id: str
    name: str
    is_flagged: bool = False
    budget_allocation: float = 0.0
    tools_used: List[str] = Field(default_factory=list)
    day_flagged: Optional[int] = None


class AdAuditState(State):
    """Full internal state for debugging and grading.

    Inherits ``episode_id`` and ``step_count`` from OpenEnv ``State``.
    """

    case_id: str = ""
    current_day: int = 0

    publishers: List[PublisherState] = Field(default_factory=list)

    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    daily_rewards: List[float] = Field(default_factory=list)
    cumulative_reward: float = 0.0

    investigation_budget_total: int = 0
    investigation_budget_used: int = 0

    flags_submitted: List[Dict[str, Any]] = Field(default_factory=list)

    grader_inputs: Dict[str, Any] = Field(default_factory=dict)
