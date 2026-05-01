from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("AirFogSim/sumo_local_map/tripinfo.csv")
DEFAULT_VEHICLE_OUTPUT = Path("AirFogSim/sumo_local_map/unified_vehicle_trajectories.csv")
DEFAULT_UAV_OUTPUT = Path("AirFogSim/sumo_local_map/unified_uav_trajectories.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-fleet replay trajectories for LASDM semantic-topology runs.")
    parser.add_argument("--input-tripinfo", default=str(DEFAULT_INPUT))
    parser.add_argument("--vehicle-output", default=str(DEFAULT_VEHICLE_OUTPUT))
    parser.add_argument("--uav-output", default=str(DEFAULT_UAV_OUTPUT))
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--x-min", type=float, default=5150.0)
    parser.add_argument("--x-max", type=float, default=7150.0)
    parser.add_argument("--y-min", type=float, default=5300.0)
    parser.add_argument("--y-max", type=float, default=7300.0)
    args = parser.parse_args()

    bounds = {
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "y_min": float(args.y_min),
        "y_max": float(args.y_max),
    }
    times = list(range(0, int(args.duration) + 1))
    source = _read_vehicle_tripinfo(Path(args.input_tripinfo))
    vehicle_rows = _build_vehicle_rows(source, int(args.vehicles), times, bounds)
    uav_rows = _build_uav_rows(int(args.uavs), times, bounds)
    _write_semicolon_csv(Path(args.vehicle_output), vehicle_rows, [
        "vehicle_id",
        "data_timestep",
        "vehicle_x",
        "vehicle_y",
        "vehicle_speed",
        "vehicle_angle",
        "vehicle_route",
    ])
    _write_semicolon_csv(Path(args.uav_output), uav_rows, [
        "uav_id",
        "data_timestep",
        "uav_x",
        "uav_y",
        "uav_z",
        "uav_speed",
        "uav_angle",
        "uav_phi",
    ])
    print(
        {
            "vehicle_rows": len(vehicle_rows),
            "uav_rows": len(uav_rows),
            "vehicle_output": str(args.vehicle_output),
            "uav_output": str(args.uav_output),
        }
    )


def _read_vehicle_tripinfo(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";")
    required = {
        "vehicle_id",
        "data_timestep",
        "vehicle_x",
        "vehicle_y",
        "vehicle_speed",
        "vehicle_angle",
        "vehicle_route",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df.dropna(subset=list(required)).sort_values(["vehicle_id", "data_timestep"]).reset_index(drop=True)


def _build_vehicle_rows(df: pd.DataFrame, vehicle_count: int, times: Sequence[int], bounds: Dict[str, float]) -> List[Dict[str, object]]:
    grouped = []
    for _vehicle_id, group in df.groupby("vehicle_id", sort=False):
        if len(group) >= 4:
            grouped.append(group.sort_values("data_timestep"))
    grouped.sort(key=len, reverse=True)
    if not grouped:
        raise ValueError("input tripinfo has no usable vehicle trajectories")
    rows: List[Dict[str, object]] = []
    for index in range(vehicle_count):
        group = grouped[index % len(grouped)]
        offset = index * 7
        route = str(group["vehicle_route"].iloc[0])
        base_times = group["data_timestep"].to_numpy(dtype=float)
        span = max(1.0, float(base_times[-1] - base_times[0]))
        for timestep in times:
            source_t = float(base_times[0]) + ((float(timestep + offset) % span))
            x = float(np.interp(source_t, base_times, group["vehicle_x"].to_numpy(dtype=float)))
            y = float(np.interp(source_t, base_times, group["vehicle_y"].to_numpy(dtype=float)))
            speed = float(np.interp(source_t, base_times, group["vehicle_speed"].to_numpy(dtype=float)))
            angle = float(np.interp(source_t, base_times, group["vehicle_angle"].to_numpy(dtype=float)))
            rows.append(
                {
                    "vehicle_id": f"vehicle_{index}",
                    "data_timestep": int(timestep),
                    "vehicle_x": round(_clip(x, bounds["x_min"], bounds["x_max"]), 3),
                    "vehicle_y": round(_clip(y, bounds["y_min"], bounds["y_max"]), 3),
                    "vehicle_speed": round(max(0.0, speed), 3),
                    "vehicle_angle": round(angle % 360.0, 3),
                    "vehicle_route": route,
                }
            )
    return rows


def _build_uav_rows(uav_count: int, times: Sequence[int], bounds: Dict[str, float]) -> List[Dict[str, object]]:
    centers = [
        (5300.0, 5500.0),
        (7000.0, 5500.0),
        (5300.0, 7100.0),
        (7000.0, 7100.0),
    ]
    rows: List[Dict[str, object]] = []
    for index in range(uav_count):
        center_x, center_y = centers[index % len(centers)]
        radius_x = 180.0 + 20.0 * (index % 5)
        radius_y = 160.0 + 15.0 * (index % 4)
        period = 80.0 + 4.0 * index
        last_position = None
        for timestep in times:
            theta = 2.0 * math.pi * ((float(timestep) / period) + (index % 5) / 5.0)
            x = _clip(center_x + radius_x * math.cos(theta), bounds["x_min"], bounds["x_max"])
            y = _clip(center_y + radius_y * math.sin(theta), bounds["y_min"], bounds["y_max"])
            z = 120.0 + 4.0 * index + 10.0 * math.sin(theta / 2.0)
            if last_position is None:
                speed = 20.0 + float(index % 8)
                angle = math.degrees(theta + math.pi / 2.0) % 360.0
            else:
                dx = x - last_position[0]
                dy = y - last_position[1]
                dz = z - last_position[2]
                speed = math.sqrt(dx * dx + dy * dy + dz * dz)
                angle = math.degrees(math.atan2(dy, dx)) % 360.0
            last_position = (x, y, z)
            rows.append(
                {
                    "uav_id": f"UAV_{index}",
                    "data_timestep": int(timestep),
                    "uav_x": round(x, 3),
                    "uav_y": round(y, 3),
                    "uav_z": round(z, 3),
                    "uav_speed": round(max(0.0, speed), 3),
                    "uav_angle": round(angle, 3),
                    "uav_phi": 0.0,
                }
            )
    return rows


def _clip(value: float, lower: float, upper: float) -> float:
    return min(float(upper), max(float(lower), float(value)))


def _write_semicolon_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
