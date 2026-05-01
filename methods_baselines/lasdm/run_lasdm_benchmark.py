from __future__ import annotations

import argparse
import json
import os
import sys


METHOD_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.baselines import baseline_names
from airfogsim.lasdm.benchmark_adapter import BENCHMARK_MODES, run_lasdm_benchmark_suite


DEFAULT_CONFIG_PATH = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LASDM AirFogSim benchmark adapter.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline", nargs="+", choices=sorted(baseline_names()), default=None)
    parser.add_argument("--seed", nargs="+", type=int, default=None)
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--mode", choices=BENCHMARK_MODES, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    exit_code, payload = run_lasdm_benchmark_suite(
        config_path=args.config,
        baseline_override=args.baseline,
        seed_override=args.seed,
        scenario_override=args.scenario,
        output_root_override=args.output_root,
        mode_override=args.mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
