import argparse
import json
import os
import sys

BENCHMARK_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BENCHMARK_ROOT, "../../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, BENCHMARK_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_runner import (
    BASELINE_REGISTRY,
    DEFAULT_CONFIG_PATH,
    run_benchmark_suite,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline", choices=sorted(BASELINE_REGISTRY.keys()), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    exit_code, payload = run_benchmark_suite(
        config_path=args.config,
        baseline_override=args.baseline,
        seed_override=args.seed,
        output_root_override=args.output_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
