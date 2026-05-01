from __future__ import annotations

import json
import os
import sys


METHOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm import build_manager_from_yaml

CONFIG_PATH = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")


def main() -> None:
    manager = build_manager_from_yaml(CONFIG_PATH)
    decisions = manager.step(current_time=0.0)
    for decision in decisions:
        if decision.accepted:
            manager.complete(decision.sfc_id, current_time=1.7)
    print(json.dumps(manager.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
