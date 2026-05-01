import unittest

from airfogsim.service_orchestration.service_catalog import ServiceCatalog
from airfogsim.service_orchestration.service_matcher import RuleBasedServiceMatcher
from airfogsim.service_orchestration.service_spec import PayloadProfile, QoSProfile, ServiceCategory, ServiceIntent


class ServiceMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        cls.catalog = ServiceCatalog.from_yaml(os.path.join(root, "examples", "service_catalog.yaml"))

    def test_linear_inspection_graph(self):
        matcher = RuleBasedServiceMatcher(self.catalog)
        intent = ServiceIntent(
            intent_id="inspection",
            source_node_id="UAV_0",
            category=ServiceCategory.MISSION_APPLICATION,
            required_capabilities=["video_preprocess", "lightweight_inference", "object_verification"],
            payload=PayloadProfile("video", 4.0),
            qos=QoSProfile(2.0),
            context={"risk_level": 0.2},
            sink_node_id="UAV_0",
        )
        graph = matcher.match(intent)
        self.assertEqual(
            graph.edges,
            [
                ("keyframe_extraction", "lightweight_object_detection"),
                ("lightweight_object_detection", "edge_object_verification"),
            ],
        )

    def test_risk_inserts_conflict_prediction(self):
        matcher = RuleBasedServiceMatcher(self.catalog)
        intent = ServiceIntent(
            intent_id="risk",
            source_node_id="UAV_0",
            category=ServiceCategory.AIRSPACE_GOVERNANCE,
            required_capabilities=["remote_id", "route_compliance"],
            payload=PayloadProfile("telemetry", 0.2),
            qos=QoSProfile(1.0),
            context={"risk_level": 0.95},
            sink_node_id="UAV_0",
        )
        graph = matcher.match(intent)
        self.assertIn("conflict_prediction", graph.nodes)
        self.assertIn(("route_compliance_check", "conflict_prediction"), graph.edges)

    def test_fixed_sfc_ignores_risk_insertion(self):
        matcher = RuleBasedServiceMatcher(self.catalog, policy="fixed_sfc")
        intent = ServiceIntent(
            intent_id="fixed",
            source_node_id="UAV_0",
            category=ServiceCategory.AIRSPACE_GOVERNANCE,
            required_capabilities=["remote_id"],
            payload=PayloadProfile("telemetry", 0.1),
            qos=QoSProfile(1.0),
            context={"risk_level": 1.0},
            sink_node_id="UAV_0",
        )
        graph = matcher.match(intent)
        self.assertNotIn("conflict_prediction", graph.nodes)


if __name__ == "__main__":
    unittest.main()
