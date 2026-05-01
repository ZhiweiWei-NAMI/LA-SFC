import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceSelectionQuery
from airfogsim.lasdm.instance_lifecycle import (
    InstanceLifecyclePhase,
    InstanceLifecycleUpdate,
    LASDMInstanceLifecycle,
    instance_qos_metadata,
)


def make_instance(instance_id="inst_0", reliability_score=0.95, trust_score=0.95):
    return ServiceInstance(
        instance_id=instance_id,
        service_id="detect",
        node_id=f"node_{instance_id}",
        node_type="rsu",
        region_id="region_0",
        capabilities=("object_detection",),
        input_semantic="keyframes",
        output_semantic="detection_result",
        capacity={"cpu": 4.0, "memory": 1024.0},
        max_concurrency=2,
        health_score=0.95,
        reliability_score=reliability_score,
        accuracy_score=0.95,
        trust_score=trust_score,
    )


def make_query(min_reliability=0.9, min_trust=0.9):
    return ServiceSelectionQuery(
        service_id="detect",
        required_capabilities=("object_detection",),
        input_semantic="keyframes",
        min_reliability=min_reliability,
        min_accuracy=0.9,
        min_trust=min_trust,
        resource_request={"cpu": 1.0, "memory": 128.0},
    )


class LASDMInstanceLifecycleTests(unittest.TestCase):
    def test_cold_start_instance_is_not_selectable_until_active(self):
        directory = ServiceInstanceDirectory()
        instance = directory.register(make_instance())
        lifecycle = LASDMInstanceLifecycle(directory)

        event = lifecycle.cold_start(instance, current_time=0.5, cold_start_s=0.4, reason="deployment")
        metadata = instance_qos_metadata(instance, {"cpu": 1.0, "memory": 128.0})

        self.assertEqual(directory.candidates(make_query()), [])
        self.assertEqual(instance.status.value, "starting")
        self.assertEqual(instance.metadata["lifecycle_phase"], "cold_start")
        self.assertEqual(instance.cold_start_s, 0.4)
        self.assertFalse(metadata["selectable"])
        self.assertEqual(metadata["blocked_reason"], "status:cold_start")
        self.assertEqual(event.payload["lifecycle_phase"], "cold_start")
        self.assertFalse(event.payload["selectable"])

    def test_starting_instance_is_not_a_candidate_and_emits_payload(self):
        directory = ServiceInstanceDirectory()
        instance = directory.register(make_instance())
        lifecycle = LASDMInstanceLifecycle(directory)

        event = lifecycle.start(instance, current_time=1.0, cold_start_s=0.25, reason="scale_up")

        self.assertEqual(directory.candidates(make_query()), [])
        self.assertEqual(instance.status.value, "starting")
        self.assertEqual(instance.metadata["lifecycle_phase"], "starting")
        self.assertEqual(instance.cold_start_s, 0.25)
        self.assertEqual(event.event_type, "instance_lifecycle_updated")
        self.assertEqual(event.payload["previous_status"], "active")
        self.assertEqual(event.payload["status"], "starting")
        self.assertEqual(event.payload["reason"], "scale_up")
        self.assertFalse(event.payload["selectable"])

    def test_active_instance_restores_candidate_selection_and_qos_metadata(self):
        directory = ServiceInstanceDirectory()
        instance = directory.register(make_instance())
        lifecycle = LASDMInstanceLifecycle(directory)
        lifecycle.start(instance, current_time=1.0, cold_start_s=0.25)

        event = lifecycle.activate(
            instance,
            current_time=1.3,
            health_score=0.9,
            reliability_score=0.94,
            trust_score=0.93,
        )

        candidates = directory.candidates(make_query())
        metadata = instance_qos_metadata(instance, {"cpu": 1.0, "memory": 128.0})
        self.assertEqual(candidates, [instance])
        self.assertEqual(instance.status.value, "active")
        self.assertEqual(instance.cold_start_s, 0.0)
        self.assertTrue(metadata["selectable"])
        self.assertIsNone(metadata["blocked_reason"])
        self.assertEqual(event.payload["previous_lifecycle_phase"], "starting")
        self.assertEqual(event.payload["lifecycle_phase"], "active")

    def test_degraded_instance_updates_scores_and_can_be_filtered_by_qos(self):
        directory = ServiceInstanceDirectory()
        healthy = directory.register(make_instance("healthy", reliability_score=0.98, trust_score=0.99))
        degraded = directory.register(make_instance("degraded", reliability_score=0.97, trust_score=0.98))
        lifecycle = LASDMInstanceLifecycle(directory)

        lifecycle.degrade(
            degraded,
            current_time=2.0,
            health_score=0.4,
            reliability_score=0.75,
            trust_score=0.7,
            reason="heartbeat_loss",
        )

        self.assertEqual(degraded.status.value, "degraded")
        self.assertEqual(directory.candidates(make_query(min_reliability=0.9, min_trust=0.9)), [healthy])
        relaxed_query = make_query(min_reliability=0.0, min_trust=0.0)
        self.assertIn(degraded, directory.candidates(relaxed_query))
        self.assertLess(
            directory.score(degraded, relaxed_query),
            directory.score(healthy, relaxed_query),
        )

    def test_draining_migrating_and_failed_instances_are_not_candidates(self):
        directory = ServiceInstanceDirectory()
        draining = directory.register(make_instance("draining"))
        migrating = directory.register(make_instance("migrating"))
        failed = directory.register(make_instance("failed"))
        lifecycle = LASDMInstanceLifecycle(directory)

        drain_event = lifecycle.drain(draining, current_time=3.0, reason="scale_down")
        migrate_event = lifecycle.migrate(
            migrating,
            current_time=3.1,
            target_node_id="node_target",
            reason="region_rebalance",
        )
        fail_event = lifecycle.fail(failed, current_time=3.2, reason="node_failure")

        self.assertEqual(directory.candidates(make_query()), [])
        self.assertEqual(draining.status.value, "draining")
        self.assertEqual(migrating.status.value, "draining")
        self.assertEqual(migrating.metadata["lifecycle_phase"], "migrating")
        self.assertEqual(migrate_event.payload["migration_target_node_id"], "node_target")
        self.assertEqual(failed.status.value, "failed")
        self.assertEqual(failed.health_score, 0.0)
        self.assertEqual(drain_event.payload["reason"], "scale_down")
        self.assertEqual(fail_event.payload["reason"], "node_failure")
        self.assertEqual(instance_qos_metadata(migrating)["blocked_reason"], "status:migrating")
        self.assertEqual(instance_qos_metadata(migrating)["migration_target_node_id"], "node_target")

    def test_recovered_instance_restores_candidates_and_qos_scores(self):
        directory = ServiceInstanceDirectory()
        instance = directory.register(make_instance())
        lifecycle = LASDMInstanceLifecycle(directory)

        lifecycle.fail(instance, current_time=4.0, reason="node_fault")
        event = lifecycle.recover(
            instance,
            current_time=4.5,
            health_score=0.88,
            reliability_score=0.93,
            accuracy_score=0.92,
            trust_score=0.91,
            reason="node_restored",
        )

        self.assertEqual(directory.candidates(make_query()), [instance])
        self.assertEqual(instance.status.value, "active")
        self.assertEqual(instance.metadata["lifecycle_phase"], "recovered")
        self.assertEqual(event.payload["previous_lifecycle_phase"], "failed")
        self.assertEqual(event.payload["lifecycle_phase"], "recovered")
        self.assertTrue(instance_qos_metadata(instance)["selectable"])
        self.assertEqual(instance.accuracy_score, 0.92)

    def test_step_updates_change_load_selectability_and_qos_metadata(self):
        directory = ServiceInstanceDirectory()
        instance = directory.register(make_instance())
        lifecycle = LASDMInstanceLifecycle(directory)

        events = lifecycle.apply_step_updates(
            {
                "inst_0": InstanceLifecycleUpdate(
                    phase=InstanceLifecyclePhase.ACTIVE,
                    current_load=2,
                    used={"cpu": 2.0, "memory": 512.0},
                    reliability_score=0.94,
                )
            },
            current_time=5.0,
        )

        metadata = instance_qos_metadata(instance, {"cpu": 1.0, "memory": 128.0})
        self.assertEqual(len(events), 1)
        self.assertEqual(instance.current_load, 2)
        self.assertEqual(instance.used["cpu"], 2.0)
        self.assertEqual(directory.candidates(make_query()), [])
        self.assertFalse(metadata["selectable"])
        self.assertEqual(metadata["blocked_reason"], "capacity:concurrency")
        self.assertEqual(metadata["current_load"], 2)
        self.assertEqual(metadata["max_concurrency"], 2)
        self.assertEqual(events[0].payload["load_ratio"], 1.0)
        self.assertFalse(events[0].payload["selectable"])


if __name__ == "__main__":
    unittest.main()
