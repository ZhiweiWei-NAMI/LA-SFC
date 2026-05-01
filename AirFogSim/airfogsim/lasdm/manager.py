from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .instance_directory import ServiceInstanceDirectory
from .metrics import LASDMMetrics
from .model import GraphStatus, LASDMServiceChain, SFCFailureReason
from .orchestrator import LASDMDecision, LASDMOrchestrator


class LASDMManager:
    """Owns LASDM SFC lifecycle, orchestration decisions, and experiment metrics."""

    def __init__(
        self,
        directory: Optional[ServiceInstanceDirectory] = None,
        orchestrator: Optional[LASDMOrchestrator] = None,
        metrics: Optional[LASDMMetrics] = None,
    ):
        self.directory = directory or ServiceInstanceDirectory()
        self.metrics = metrics or LASDMMetrics()
        self.orchestrator = orchestrator or LASDMOrchestrator(self.directory)
        self.chains: Dict[str, LASDMServiceChain] = {}
        self.decisions: Dict[str, LASDMDecision] = {}

    def submit(self, chain: LASDMServiceChain, current_time: float = 0.0) -> LASDMServiceChain:
        chain.validate()
        chain.mark_running(current_time)
        self.chains[chain.sfc_id] = chain
        self.metrics.record_submit(chain.sfc_id, current_time)
        return chain

    def submit_many(self, chains: Iterable[LASDMServiceChain], current_time: float = 0.0) -> None:
        for chain in chains:
            self.submit(chain, current_time)

    def plan_pending(self) -> List[LASDMDecision]:
        decisions: List[LASDMDecision] = []
        for chain in self.chains.values():
            if chain.status != GraphStatus.RUNNING or chain.sfc_id in self.decisions:
                continue
            decision = self.orchestrator.plan(chain)
            self.decisions[chain.sfc_id] = decision
            decisions.append(decision)
        return decisions

    def complete(self, sfc_id: str, current_time: float) -> None:
        chain = self.chains[sfc_id]
        if chain.is_terminal():
            return
        submit_time = chain.submit_time if chain.submit_time is not None else current_time
        chain.mark_succeeded(current_time)
        qos_hit = current_time - submit_time <= chain.qos.deadline_s
        self.metrics.record_success(sfc_id, submit_time, current_time, qos_hit=qos_hit)

    def fail(
        self,
        sfc_id: str,
        reason: SFCFailureReason,
        current_time: float,
        details: Optional[Dict[str, object]] = None,
        status: GraphStatus = GraphStatus.FAILED,
    ) -> None:
        chain = self.chains[sfc_id]
        if chain.is_terminal():
            return
        if status == GraphStatus.TIMED_OUT:
            chain.mark_timed_out(current_time)
        else:
            chain.mark_failed(reason, current_time)
        self.metrics.record_failure(sfc_id, current_time, reason, status=status, details=details)

    def step(self, current_time: float) -> List[LASDMDecision]:
        for chain in self.chains.values():
            if chain.status != GraphStatus.RUNNING or chain.submit_time is None:
                continue
            if current_time - chain.submit_time > chain.qos.deadline_s:
                chain.mark_timed_out(current_time)
                self.metrics.record_failure(
                    chain.sfc_id,
                    current_time,
                    SFCFailureReason.DEADLINE_MISSED,
                    status=GraphStatus.TIMED_OUT,
                )
        decisions = self.plan_pending()
        for decision in decisions:
            if decision.rejected_reason is not None:
                self.fail(decision.sfc_id, decision.rejected_reason, current_time)
        return decisions

    def summary(self) -> Dict[str, object]:
        summary = self.metrics.summary()
        summary["active_graphs"] = len([chain for chain in self.chains.values() if chain.status == GraphStatus.RUNNING])
        summary["decisions"] = {sfc_id: decision.to_dict() for sfc_id, decision in self.decisions.items()}
        return summary
