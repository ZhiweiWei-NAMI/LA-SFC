"""LASDM declarative service orchestration extensions for AirFogSim.

This initializer exposes the runtime-integrated LASDM and semantic-topology MARL
modules. Missing runtime dependencies are treated as configuration errors.
"""

from .instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceInstanceStatus, ServiceSelectionQuery
from .manager import LASDMManager
from .metrics import LASDMMetrics, LASDMEvent
from .model import GraphStatus, LASDMQoS, LASDMServiceChain, LASDMSFCNode, SFCFailureReason
from .orchestrator import LASDMDecision, LASDMOrchestrator
from .marl import LASDMMARLInterface
from .api import build_manager_from_yaml, load_chains_from_yaml, load_instances_from_yaml

# Semantic-topology MARL extension exports.
from .semantic_encoder import SemanticEncoder, SemanticTextRecord, SemanticMatch
from .semantic_cache import EmbeddingCache, SemanticAdvertisement, SemanticAdvertisementCache
from .semantic_exchange import SemanticExchange, SemanticExchangeConfig
from .distributed_catalog import CatalogCandidate, DistributedServiceCatalog
from .service_discovery_protocol import DistributedServiceDiscoveryProtocol, DiscoveryRequest
from .topology_builder import DynamicTopology, TopologyBuilder, TopologyNode, TopologyEdge
from .temporal_state_buffer import TemporalStateBuffer, TemporalSummary
from .graph_observation import GraphObservationBuilder, GraphObservationConfig, flatten_observation
from .marl_env import MARLEnvConfig, SemanticTopologyMARLEnv
from .marl_policy import IPPOPolicy, MASACPolicy, SemanticGreedyPolicy, TopologyGreedyPolicy, RandomValidPolicy
from .marl_reward import SFCReward, SFCRewardConfig, compute_sfc_reward

from .env_adapter import LASDMEnvAdapter
from .runtime_bridge import LASDMRuntimeBridge
from .scheduler import LASDMScheduler
from .task_adapter import LASDMTaskAdapter, LASDMTaskMapping, build_lasdm_task, propagate_parent_failure

__all__ = [name for name in globals() if not name.startswith("_")]
