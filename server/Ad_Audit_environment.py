"""
AdAuditEnv — main environment class.

Wires together publisher_engine, fraud_engine, response_generator,
step_reward, and grader into the OpenEnv Environment interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AdAuditAction,
        AdAuditObservation,
        AdAuditState,
        BudgetStatus,
        DailyPublisherMetrics,
        PublisherState,
    )
    from .fraud_engine import (
        decay_suspicion,
        get_adaptation_stage,
        update_suspicion,
    )
    from .publisher_engine import generate_daily_traffic
    from .response_generator import (
        generate_alerts,
        generate_investigation_metrics,
        generate_trend_summary,
    )
    from .step_reward import compute_step_reward
    from .grader import grade_episode
except ImportError:
    from models import (  # type: ignore[no-redef]
        AdAuditAction,
        AdAuditObservation,
        AdAuditState,
        BudgetStatus,
        DailyPublisherMetrics,
        PublisherState,
    )
    from server.fraud_engine import (  # type: ignore[no-redef]
        decay_suspicion,
        get_adaptation_stage,
        update_suspicion,
    )
    from server.publisher_engine import generate_daily_traffic  # type: ignore[no-redef]
    from server.response_generator import (  # type: ignore[no-redef]
        generate_alerts,
        generate_investigation_metrics,
        generate_trend_summary,
    )
    from server.step_reward import compute_step_reward  # type: ignore[no-redef]
    from server.grader import grade_episode  # type: ignore[no-redef]

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"

TASK_MAP = {
    "easy": "easy.json",
    "medium": "medium.json",
    "hard": "hard.json",
}

EPISODE_DAYS = 14


class _PubInternal:
    """Hidden per-publisher state (not exposed via /state)."""
    __slots__ = (
        "is_fraudulent", "fraud_type", "suspicion_level", "adaptation_stage",
        "total_fraudulent_spend", "total_legitimate_spend", "total_legitimate_revenue",
    )

    def __init__(self, is_fraudulent: bool = False, fraud_type: str = None):
        self.is_fraudulent = is_fraudulent
        self.fraud_type = fraud_type
        self.suspicion_level = 0.0
        self.adaptation_stage = "normal"
        self.total_fraudulent_spend = 0.0
        self.total_legitimate_spend = 0.0
        self.total_legitimate_revenue = 0.0


class AdAuditEnv(Environment[AdAuditAction, AdAuditObservation, AdAuditState]):
    """OpenEnv-compatible RL environment for ad fraud detection."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    @classmethod
    def get_tasks(cls) -> List[str]:
        return list(TASK_MAP.keys())

    _TASK_CYCLE = ["easy", "medium", "hard"]

    def __init__(self) -> None:
        super().__init__()
        self._case: Dict[str, Any] = {}
        self._state = AdAuditState()
        self._pub_cfgs: Dict[str, Dict[str, Any]] = {}
        self._pub_names: Dict[str, str] = {}
        self._pub_internal: Dict[str, _PubInternal] = {}
        self._daily_logs: Dict[str, List[Dict[str, Any]]] = {}
        self._step_action: Optional[AdAuditAction] = None
        self._cycle_index: int = 0
        self._invalid_action: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AdAuditObservation:
        task_id = kwargs.get("task_id") or episode_id
        if not task_id:
            task_id = self._TASK_CYCLE[self._cycle_index % len(self._TASK_CYCLE)]
            self._cycle_index += 1
        case_file = CASES_DIR / TASK_MAP.get(task_id, f"{task_id}.json")
        with open(case_file) as f:
            self._case = json.load(f)

        campaign = self._case["campaign"]
        publishers = self._case["publishers"]

        pub_states: List[PublisherState] = []
        self._pub_cfgs = {}
        self._pub_names = {}
        self._pub_internal = {}
        self._daily_logs = {}

        for pub_id, cfg in publishers.items():
            self._pub_cfgs[pub_id] = cfg
            self._pub_names[pub_id] = cfg.get("name", pub_id)
            self._daily_logs[pub_id] = []

            pub_states.append(PublisherState(
                publisher_id=pub_id,
                name=cfg.get("name", pub_id),
                budget_allocation=cfg.get("budget_allocation", 1.0 / len(publishers)),
            ))

            self._pub_internal[pub_id] = _PubInternal(
                is_fraudulent=cfg.get("is_fraudulent", False),
                fraud_type=cfg.get("fraud_type"),
            )

        self._state = AdAuditState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            case_id=self._case.get("case_id", task_id),
            current_day=0,
            publishers=pub_states,
            investigation_budget_total=campaign.get("investigation_budget", 8),
            investigation_budget_used=0,
        )

        self._step_action = None
        self._invalid_action = False
        return self._advance_day()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: AdAuditAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AdAuditObservation:
        if self._state.current_day >= EPISODE_DAYS:
            return self._finalize("Campaign ended.")

        investigation_result: Optional[Dict[str, Any]] = None
        self._invalid_action = False
        at = action.action_type
        self._state.action_history.append(action.model_dump(exclude_none=True))
        self._step_action = action

        if at == "monitor":
            pass
        elif at == "investigate_publisher":
            investigation_result = self._handle_investigate(action)
            if investigation_result and "error" in investigation_result:
                self._invalid_action = True
        elif at == "flag_fraud":
            ps = self._get_pub_state(action.publisher_id)
            if ps is None or ps.is_flagged or not action.fraud_type:
                self._invalid_action = True
            else:
                self._handle_flag_fraud(action)
        elif at == "submit_report":
            return self._finalize("Agent submitted report.")
        else:
            self._invalid_action = True

        return self._advance_day(investigation_result=investigation_result)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> AdAuditState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_investigate(self, action: AdAuditAction) -> Optional[Dict[str, Any]]:
        pub_id = action.publisher_id
        tool = action.tool
        if not pub_id or not tool:
            return {"error": "publisher_id and tool are required"}

        ps = self._get_pub_state(pub_id)
        if ps is None:
            valid = [p.publisher_id for p in self._state.publishers]
            return {"error": f"unknown publisher_id: {pub_id}. Valid IDs: {valid}"}

        if ps.is_flagged:
            return {"error": f"{pub_id} is already flagged."}

        budget_remaining = (
            self._state.investigation_budget_total
            - self._state.investigation_budget_used
        )
        if budget_remaining <= 0:
            return {"error": "no investigation budget remaining"}

        self._state.investigation_budget_used += 1

        cfg = self._pub_cfgs.get(pub_id, {})
        hi = self._pub_internal[pub_id]

        if tool not in ps.tools_used:
            ps.tools_used.append(tool)

        if hi.is_fraudulent:
            hi.suspicion_level = update_suspicion(
                hi.suspicion_level, tool, cfg.get("suspicion_reactivity", 1.0),
            )
            hi.adaptation_stage = get_adaptation_stage(hi.suspicion_level)

        return generate_investigation_metrics(
            case_id=self._state.case_id,
            publisher_id=pub_id,
            publisher_cfg=cfg,
            tool_name=tool,
            adaptation_stage=hi.adaptation_stage,
        )

    def _handle_flag_fraud(self, action: AdAuditAction) -> None:
        pub_id = action.publisher_id
        ps = self._get_pub_state(pub_id)
        hi = self._pub_internal[pub_id]

        ps.is_flagged = True
        ps.day_flagged = self._state.current_day + 1

        is_correct = hi.is_fraudulent
        type_correct = (action.fraud_type == hi.fraud_type) if is_correct else False

        self._state.flags_submitted.append({
            "publisher_id": pub_id,
            "fraud_type": action.fraud_type,
            "evidence": action.evidence or [],
            "day": self._state.current_day + 1,
            "correct": is_correct,
            "type_correct": type_correct,
        })

    # ------------------------------------------------------------------
    # Day advancement
    # ------------------------------------------------------------------

    def _advance_day(
        self,
        investigation_result: Optional[Dict[str, Any]] = None,
    ) -> AdAuditObservation:
        self._state.current_day += 1
        self._state.step_count = self._state.current_day
        day = self._state.current_day

        # Decay suspicion for publishers NOT investigated today
        investigated_today = set()
        if self._state.action_history:
            last = self._state.action_history[-1]
            if last.get("action_type") == "investigate_publisher" and last.get("publisher_id"):
                investigated_today.add(last["publisher_id"])

        for ps in self._state.publishers:
            hi = self._pub_internal[ps.publisher_id]
            if hi.is_fraudulent and ps.publisher_id not in investigated_today:
                hi.suspicion_level = decay_suspicion(hi.suspicion_level)
                hi.adaptation_stage = get_adaptation_stage(hi.suspicion_level)

        # Generate traffic
        daily_traffic: List[Dict[str, Any]] = []
        daily_metrics: List[DailyPublisherMetrics] = []
        campaign = self._case["campaign"]
        benchmarks = campaign.get("industry_benchmarks", {})

        daily_fraud_spend = 0.0
        for ps in self._state.publishers:
            cfg = self._pub_cfgs.get(ps.publisher_id, {})
            hi = self._pub_internal[ps.publisher_id]
            traffic = generate_daily_traffic(
                day=day, publisher_cfg=cfg,
                budget_allocation=ps.budget_allocation,
                adaptation_stage=hi.adaptation_stage,
                is_paused=ps.is_flagged,
            )
            daily_traffic.append(traffic)
            self._daily_logs[ps.publisher_id].append(traffic)

            hi.total_legitimate_spend += traffic["legitimate_spend"]
            hi.total_fraudulent_spend += traffic["fraudulent_spend"]
            hi.total_legitimate_revenue += traffic["legitimate_revenue"]

            if hi.is_fraudulent and not ps.is_flagged:
                daily_fraud_spend += traffic["fraudulent_spend"]

            daily_metrics.append(DailyPublisherMetrics(
                publisher_id=ps.publisher_id, name=ps.name,
                impressions=traffic["impressions"], clicks=traffic["clicks"],
                conversions=traffic["conversions"], spend=traffic["spend"],
                ctr=traffic["ctr"], cvr=traffic["cvr"],
            ))

        # --- Compute step reward ---
        action = self._step_action
        total_budget = campaign["total_budget"]

        if action is None:
            step_reward = 0.0
        elif self._invalid_action:
            step_reward = compute_step_reward(
                action_type="invalid",
                daily_fraud_spend=daily_fraud_spend,
                total_budget=total_budget,
                day=day,
                episode_days=EPISODE_DAYS,
            )
        elif action.action_type == "flag_fraud":
            last_flag = self._state.flags_submitted[-1] if self._state.flags_submitted else {}
            step_reward = compute_step_reward(
                action_type="flag_fraud",
                flag_correct=last_flag.get("correct"),
                flag_type_correct=last_flag.get("type_correct"),
                daily_fraud_spend=daily_fraud_spend,
                total_budget=total_budget,
                day=day,
                episode_days=EPISODE_DAYS,
            )
        elif action.action_type == "investigate_publisher":
            pub_cfg = self._pub_cfgs.get(action.publisher_id, {})
            step_reward = compute_step_reward(
                action_type="investigate_publisher",
                publisher_cfg=pub_cfg,
                daily_fraud_spend=daily_fraud_spend,
                total_budget=total_budget,
                day=day,
                episode_days=EPISODE_DAYS,
            )
        else:
            step_reward = compute_step_reward(
                action_type=action.action_type,
                daily_fraud_spend=daily_fraud_spend,
                total_budget=total_budget,
                day=day,
                episode_days=EPISODE_DAYS,
            )

        # Trend + alerts
        trend_data = ""  # TODO: generate_trend_summary(self._daily_logs, self._pub_names, day)
        raw_metrics = [
            {"publisher_id": m.publisher_id, "ctr": m.ctr, "cvr": m.cvr,
             "impressions": m.impressions, "clicks": m.clicks}
            for m in daily_metrics
        ]
        alerts = []  # TODO: generate_alerts(raw_metrics, benchmarks, self._pub_names)

        cumulative_metrics = self._compute_cumulative_metrics()

        total_spend = sum(
            hi.total_legitimate_spend + hi.total_fraudulent_spend
            for hi in self._pub_internal.values()
        )
        budget_status = BudgetStatus(
            total_campaign_budget=campaign["total_budget"],
            spent_so_far=round(total_spend, 2),
            remaining=round(campaign["total_budget"] - total_spend, 2),
            investigation_budget_remaining=(
                self._state.investigation_budget_total
                - self._state.investigation_budget_used
            ),
            daily_spend_rate=round(total_spend / day, 2) if day > 0 else 0.0,
        )

        pub_status = {
            ps.publisher_id: ("flagged" if ps.is_flagged else "active")
            for ps in self._state.publishers
        }

        # Termination
        done = False
        done_reason: Optional[str] = None

        if day >= EPISODE_DAYS:
            done = True
            done_reason = f"Campaign ended (day {EPISODE_DAYS})."
        elif budget_status.remaining <= 0:
            done = True
            done_reason = "Campaign budget exhausted."

        # On episode end, compute grader (stored separately, not in step reward)
        if done:
            self._state.grader_inputs = grade_episode(
                self._build_grader_state(), self._case,
            )

        step_reward = round(max(0.0, min(1.0, step_reward)), 4)
        self._state.daily_rewards.append(step_reward)
        self._state.cumulative_reward += step_reward

        return self._apply_transform(AdAuditObservation(
            day=day,
            campaign_day_total=EPISODE_DAYS,
            daily_metrics=daily_metrics,
            cumulative_metrics=cumulative_metrics,
            trend_data=trend_data,
            investigation_results=investigation_result,
            alerts=alerts,
            budget_status=budget_status,
            publisher_status=pub_status,
            cumulative_reward=round(self._state.cumulative_reward, 4),
            done=done,
            done_reason=done_reason,
            reward=step_reward,
        ))

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def _finalize(self, reason: str) -> AdAuditObservation:
        self._state.grader_inputs = grade_episode(
            self._build_grader_state(), self._case,
        )
        grader_score = self._state.grader_inputs.get("final_score", 0.0)

        step_reward = grader_score
        self._state.daily_rewards.append(step_reward)
        self._state.cumulative_reward += step_reward

        return AdAuditObservation(
            day=self._state.current_day,
            campaign_day_total=EPISODE_DAYS,
            trend_data=f"Episode complete. Grader score: {grader_score:.4f}",
            done=True,
            done_reason=reason,
            reward=step_reward,
            cumulative_reward=round(self._state.cumulative_reward, 4),
            publisher_status={
                ps.publisher_id: ("flagged" if ps.is_flagged else "active")
                for ps in self._state.publishers
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pub_state(self, pub_id: Optional[str]) -> Optional[PublisherState]:
        for ps in self._state.publishers:
            if ps.publisher_id == pub_id:
                return ps
        return None

    def _build_grader_state(self) -> Dict[str, Any]:
        """Build the state dict the grader expects, including hidden fields."""
        state_dict = self._state.model_dump()
        # Enrich publisher entries with hidden internal state for grading
        for pub_dict in state_dict["publishers"]:
            pub_id = pub_dict["publisher_id"]
            hi = self._pub_internal[pub_id]
            pub_dict["is_fraudulent"] = hi.is_fraudulent
            pub_dict["fraud_type"] = hi.fraud_type
            pub_dict["suspicion_level"] = hi.suspicion_level
            pub_dict["total_fraudulent_spend"] = hi.total_fraudulent_spend
            pub_dict["total_legitimate_spend"] = hi.total_legitimate_spend
        return state_dict

    def _compute_cumulative_metrics(self) -> List[DailyPublisherMetrics]:
        result = []
        for ps in self._state.publishers:
            logs = self._daily_logs.get(ps.publisher_id, [])
            if not logs:
                continue
            total_imp = sum(d["impressions"] for d in logs)
            total_clicks = sum(d["clicks"] for d in logs)
            total_conv = sum(d["conversions"] for d in logs)
            total_spend = sum(d["spend"] for d in logs)
            ctr = total_clicks / total_imp if total_imp > 0 else 0.0
            cvr = total_conv / total_clicks if total_clicks > 0 else 0.0
            result.append(DailyPublisherMetrics(
                publisher_id=ps.publisher_id, name=ps.name,
                impressions=total_imp, clicks=total_clicks,
                conversions=total_conv, spend=round(total_spend, 2),
                ctr=round(ctr, 4), cvr=round(cvr, 4),
            ))
        return result


# Alias used by app.py and server/__init__.py
AdAuditEnvironment = AdAuditEnv
