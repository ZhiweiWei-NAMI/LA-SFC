import argparse
import os
import random
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airfogsim import AirFogSimEnv
from airfogsim.service_orchestration.agentic_service_algorithm import AgenticServiceAlgorithmModule
from airfogsim.service_orchestration.testing import advance_algorithm_until_complete, build_smoke_env
from examples.task_intents import PoissonLowAltitudeIntentGenerator, StaticLowAltitudeIntentGenerator


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_smoke_test(catalog_path: str, intents_path: str) -> int:
    env = build_smoke_env(multi_region=True)
    intent_generator = StaticLowAltitudeIntentGenerator.from_yaml(intents_path)
    algorithm = AgenticServiceAlgorithmModule(
        catalog_path=catalog_path,
        intent_generator=intent_generator,
        regional_agent_enabled=True,
    )
    algorithm.initialize(env)
    ok = advance_algorithm_until_complete(algorithm, env, max_rounds=12)
    if not ok:
        print("Smoke test failed: service graph did not finish.")
        return 1
    print("Smoke test passed.")
    print(algorithm.metrics.summary())
    return 0


def run_airfogsim_example(config_path: str, catalog_path: str, max_steps: int | None = None) -> int:
    config = load_config(config_path)
    config["task"]["task_generation_model"] = "None"

    np.random.seed(0)
    random.seed(0)

    env = AirFogSimEnv(config, interactive_mode=None)
    orchestrator_config = config.get("service_orchestration", {})
    intent_generator = PoissonLowAltitudeIntentGenerator(arrival_prob=0.3)
    algorithm = AgenticServiceAlgorithmModule(
        catalog_path=catalog_path,
        intent_generator=intent_generator,
        matcher_policy=orchestrator_config.get("matcher_policy", "adaptive_rule_based"),
        serving_policy=orchestrator_config.get("serving_policy", "proposed"),
        lazy_deploy=orchestrator_config.get("lazy_deploy", True),
        regional_agent_enabled=orchestrator_config.get("regional_agent_enabled", False),
    )
    algorithm.initialize(env, config)

    steps = 0
    while not env.isDone():
        if max_steps is not None and steps >= max_steps:
            break
        algorithm.scheduleStep(env)
        env.step()
        succ = algorithm.taskScheduler.getDoneTaskNum(env)
        fail = algorithm.taskScheduler.getOutOfDDLTasks(env)
        ratio = succ / max(1, succ + fail)
        print(
            f"time={env.simulation_time:.2f}, succ={succ}, fail={fail}, ratio={ratio:.3f}",
            end="\r",
        )
        steps += 1

    env.close()
    print("\nSimulation done.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_agentic_service.yaml"))
    parser.add_argument("--catalog", default=os.path.join(os.path.dirname(__file__), "service_catalog.yaml"))
    parser.add_argument("--intents", default=os.path.join(os.path.dirname(__file__), "task_intents.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        raise SystemExit(run_smoke_test(args.catalog, args.intents))
    raise SystemExit(run_airfogsim_example(args.config, args.catalog, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
