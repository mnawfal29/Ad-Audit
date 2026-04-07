# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ad Audit Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import AdAuditAction, AdAuditObservation, AdAuditState
except ImportError:
    from models import AdAuditAction, AdAuditObservation, AdAuditState  # type: ignore[no-redef]


class AdAuditEnv(
    EnvClient[AdAuditAction, AdAuditObservation, AdAuditState]
):
    """
    Client for the Ad Audit Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example with Docker:
        >>> client = await AdAuditEnv.from_docker_image("adaudit-env:latest")
        >>> try:
        ...     result = await client.reset(episode_id="medium")
        ...     result = await client.step(AdAuditAction(action_type="monitor"))
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: AdAuditAction) -> Dict[str, Any]:
        """
        Convert AdAuditAction to JSON payload for step message.

        The server deserializes this via AdAuditAction.model_validate(),
        so we just send the pydantic model_dump with None fields excluded.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdAuditObservation]:
        """
        Parse server response into StepResult[AdAuditObservation].

        The server sends:
        {
            "observation": { ... AdAuditObservation fields (minus reward/done/metadata) ... },
            "reward": float | None,
            "done": bool,
        }
        """
        obs_data = payload.get("observation", {})

        # Re-inject reward/done so the Observation model has them
        obs_data["reward"] = payload.get("reward")
        obs_data["done"] = payload.get("done", False)

        observation = AdAuditObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AdAuditState:
        """
        Parse server response into AdAuditState.
        """
        return AdAuditState.model_validate(payload)
