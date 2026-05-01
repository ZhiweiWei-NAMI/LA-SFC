from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from .env_adapter import LASDMEnvAdapter
from .manager import LASDMManager
from .model import LASDMServiceChain
from .orchestrator import LASDMDecision
from .runtime_bridge import LASDMRuntimeBridge


class LASDMScheduler:
    """AirFogSim-compatible scheduleStep facade for LASDM experiments.

    The facade prepares LASDM generated AirFogSim tasks before env.step while leaving
    AirFogSim's internal physics and queue update order unchanged.
    """

    def __init__(
        self,
        manager: LASDMManager,
        chain_provider: Optional[Callable[[object], Iterable[LASDMServiceChain]]] = None,
        runtime_bridge: Optional[LASDMRuntimeBridge] = None,
        env_adapter: Optional[LASDMEnvAdapter] = None,
    ):
        self.manager = manager
        self.chain_provider = chain_provider or (lambda env: [])
        self.env_adapter = env_adapter or LASDMEnvAdapter(directory=self.manager.directory)
        self.runtime_bridge = runtime_bridge or LASDMRuntimeBridge(self.manager, env_adapter=self.env_adapter)

    def scheduleStep(self, env) -> List[LASDMDecision]:
        current_time = float(getattr(env, "simulation_time", 0.0))
        report = self.runtime_bridge.prepare_airfogsim_step(
            env,
            chains=list(self.chain_provider(env)),
            current_time=current_time,
        )
        decisions = list(self.manager.decisions.values())
        setattr(env, "lasdm_last_decisions", [decision.to_dict() for decision in decisions])
        setattr(env, "lasdm_metrics", self.manager.summary())
        setattr(env, "lasdm_runtime_report", report)
        return decisions
