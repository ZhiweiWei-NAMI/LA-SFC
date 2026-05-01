import os
import tempfile
import unittest

from airfogsim.service_orchestration.service_catalog import ServiceCatalog
from airfogsim.service_orchestration.service_spec import ServiceCategory


ROOT = os.path.dirname(os.path.dirname(__file__))


class ServiceCatalogTests(unittest.TestCase):
    def test_load_catalog_and_categories(self):
        catalog = ServiceCatalog.from_yaml(os.path.join(ROOT, "examples", "service_catalog.yaml"))
        self.assertIn("remote_id_check", catalog.microservices)
        self.assertIn("inspection_report", catalog.microservices)
        categories = {spec.category for spec in catalog.all()}
        self.assertEqual(
            categories,
            {
                ServiceCategory.CONNECTIVITY_C2,
                ServiceCategory.AIRSPACE_GOVERNANCE,
                ServiceCategory.SENSING_SITUATION,
                ServiceCategory.TRAJECTORY_CONTROL,
                ServiceCategory.MISSION_APPLICATION,
                ServiceCategory.EDGE_INTELLIGENCE,
                ServiceCategory.CONTINUITY_RESILIENCE,
            },
        )

    def test_node_type_compatibility(self):
        catalog = ServiceCatalog.from_yaml(os.path.join(ROOT, "examples", "service_catalog.yaml"))
        edge_verifier = catalog.get("edge_object_verification")
        self.assertEqual(edge_verifier.allowed_node_types, ["rsu", "cloud_server"])

    def test_catalog_covers_required_task_intent_capabilities(self):
        import yaml

        catalog = ServiceCatalog.from_yaml(os.path.join(ROOT, "examples", "service_catalog.yaml"))
        with open(os.path.join(ROOT, "examples", "task_intents.yaml"), "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        scenarios = {item["context"]["scenario"] for item in raw["intents"]}
        self.assertEqual(
            scenarios,
            {
                "urban_monitoring",
                "logistics_governance",
                "target_tracking_search",
                "c2_continuity_emergency",
            },
        )
        for item in raw["intents"]:
            catalog.validate_required_capabilities(item["required_capabilities"])

    def test_catalog_rejects_invalid_node_type(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
            file.write(
                """
microservices:
  - ms_id: bad_node
    category: connectivity/C2
    provides: [c2_monitoring]
    cpu_per_mb: 0.1
    allowed_node_types: [balloon]
  - ms_id: governance
    category: airspace_governance
    provides: [remote_id]
    cpu_per_mb: 0.1
  - ms_id: sensing
    category: sensing/situation_awareness
    provides: [video_preprocess]
    cpu_per_mb: 0.1
  - ms_id: trajectory
    category: trajectory/cooperative_control
    provides: [trajectory_replan]
    cpu_per_mb: 0.1
  - ms_id: mission
    category: mission_oriented_application
    provides: [mission_report]
    cpu_per_mb: 0.1
  - ms_id: intelligence
    category: edge_intelligence/computing
    provides: [lightweight_inference]
    cpu_per_mb: 0.1
  - ms_id: resilience
    category: continuity/resilience
    provides: [resilience]
    cpu_per_mb: 0.1
"""
            )
            path = file.name
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        with self.assertRaisesRegex(ValueError, "Invalid allowed_node_types"):
            ServiceCatalog.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
