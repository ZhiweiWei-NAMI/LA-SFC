from __future__ import annotations

from typing import Any, Dict, Mapping

from .marl_policy import IPPOPolicy, RESOURCE_LEVEL_VALUES


class MAPPOPolicy(IPPOPolicy):
    """On-policy MAPPO-CTDE baseline over the LASDM candidate action surface."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["centralized_critic"] = True
        super().__init__(*args, **kwargs)

    def mappo_state_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "mappo_ctde_candidate_resource",
            "resource_levels": list(RESOURCE_LEVEL_VALUES),
            "actor_critic": self.model.state_dict(),
        }

    def load_mappo_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        actor_state = state.get("actor_critic", state) if isinstance(state, Mapping) else state
        self.model.load_state_dict(actor_state, strict=strict)
