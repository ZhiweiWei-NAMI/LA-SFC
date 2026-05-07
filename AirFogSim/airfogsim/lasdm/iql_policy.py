from __future__ import annotations

from typing import Any, Dict, Mapping

from .marl_policy import MASACPolicy, RESOURCE_LEVEL_VALUES


class IQLPolicy(MASACPolicy):
    """Offline IQL baseline using the MASAC actor surface plus an expectile V network."""

    def __init__(
        self,
        *args: Any,
        expectile: float = 0.7,
        beta: float = 3.0,
        v_lr: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        hidden_dim = int(getattr(self.model, "hidden_dim"))
        v_input_dim = hidden_dim * self.max_critic_agents + hidden_dim + self.candidate_feature_dim
        self.v_net = self._build_q_network(v_input_dim, hidden_dim).to(self.device)
        self.v_optimizer = self.torch.optim.Adam(
            self.v_net.parameters(),
            lr=float(v_lr if v_lr is not None else kwargs.get("q_lr", kwargs.get("lr", 3e-4))),
        )
        self.expectile = float(expectile)
        self.beta = float(beta)

    def iql_v_input(self, item: Mapping[str, Any]) -> Any:
        candidate_features = item["candidate_features"]
        if candidate_features.shape[0] <= 0:
            pooled = self.torch.zeros((self.candidate_feature_dim,), dtype=self.torch.float32, device=self.device)
        else:
            pooled = candidate_features.mean(dim=0)
        return self.torch.cat(
            [
                item["global_context"].reshape(-1),
                item["local_context"].reshape(-1),
                pooled.reshape(-1),
            ],
            dim=-1,
        )

    def iql_state_value(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        detach_encoder: bool = True,
        action_filter: Mapping[str, str] | None = None,
        action_context: Mapping[str, Any] | None = None,
    ) -> Any:
        values = []
        for item in self._masac_candidate_items(
            observations,
            target=False,
            detach_encoder=detach_encoder,
            action_filter=action_filter,
            action_context=action_context,
        ):
            values.append(self.v_net(self.iql_v_input(item)).squeeze(-1))
        if not values:
            return self.torch.tensor(0.0, dtype=self.torch.float32, device=self.device)
        return self.torch.stack([value.reshape(()) for value in values]).mean()

    def iql_state_dict(self) -> Dict[str, Any]:
        state = self.sac_state_dict()
        state.update(
            {
                "algorithm": "iql_offline_candidate_resource",
                "resource_levels": list(RESOURCE_LEVEL_VALUES),
                "v_net": self.v_net.state_dict(),
                "expectile": float(self.expectile),
                "beta": float(self.beta),
            }
        )
        return state

    def load_iql_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        self.load_sac_state_dict(state, strict=strict)
        if "v_net" in state:
            self.v_net.load_state_dict(state["v_net"], strict=strict)
        self.expectile = float(state.get("expectile", self.expectile))
        self.beta = float(state.get("beta", self.beta))
