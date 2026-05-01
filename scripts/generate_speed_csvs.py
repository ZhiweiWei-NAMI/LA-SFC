#!/usr/bin/env python3
"""Generate speed-variant vehicle trajectory CSVs by time-scaling the base trajectory.

For each speed level, vehicle positions are interpolated at a different rate:
- 50 km/h (base): use original CSV as-is
- 30 km/h: positions change slower (0.6x rate) → less topology churn
- 70 km/h: positions change faster (1.4x rate) → more topology churn
- 90 km/h: positions change fastest (1.8x rate) → most topology churn

All output CSVs have 600 timesteps, 100 vehicles, same format.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
SUMO_DIR = WORKSPACE / "AirFogSim" / "sumo_local_map"
BASE_CSV = SUMO_DIR / "unified_vehicle_trajectories.csv"

# Base average speed ~50 km/h.  Speed factors relative to base.
SPEED_FACTORS = {
    30: 0.60,
    50: 1.00,
    70: 1.40,
    90: 1.81,
}


def generate_speed_csv(speed_kmh: int, factor: float) -> None:
    output_path = SUMO_DIR / f"unified_vehicle_trajectories_{speed_kmh}kmh.csv"
    if output_path.exists():
        print(f"  [{speed_kmh} km/h] Already exists, skipping")
        return

    print(f"  [{speed_kmh} km/h] factor={factor:.2f} ...")
    df = pd.read_csv(BASE_CSV, sep=";")
    df["data_timestep"] = df["data_timestep"].astype(float)

    rows = []
    for vid, group in df.groupby("vehicle_id"):
        group = group.sort_values("data_timestep")
        times = group["data_timestep"].values.astype(float)
        xs = group["vehicle_x"].values.astype(float)
        ys = group["vehicle_y"].values.astype(float)
        speeds = group["vehicle_speed"].values.astype(float)
        angles = group["vehicle_angle"].values.astype(float)
        route = str(group["vehicle_route"].iloc[0])

        max_t = times[-1]

        for t in range(601):
            src_t = t * factor
            if src_t > max_t:
                # Vehicle has finished its route — reuse last position
                x, y, sp, ang = xs[-1], ys[-1], 0.0, angles[-1]
            else:
                idx = np.searchsorted(times, src_t)
                if idx == 0:
                    x, y, sp, ang = xs[0], ys[0], speeds[0], angles[0]
                elif idx >= len(times):
                    x, y, sp, ang = xs[-1], ys[-1], speeds[-1], angles[-1]
                else:
                    # Linear interpolation
                    t_lo, t_hi = times[idx - 1], times[idx]
                    frac = (src_t - t_lo) / max(1e-9, t_hi - t_lo)
                    x = xs[idx - 1] + frac * (xs[idx] - xs[idx - 1])
                    y = ys[idx - 1] + frac * (ys[idx] - ys[idx - 1])
                    sp = (speeds[idx - 1] + frac * (speeds[idx] - speeds[idx - 1])) * factor
                    ang = angles[idx - 1] + frac * (angles[idx] - angles[idx - 1])

            rows.append({
                "vehicle_id": vid,
                "data_timestep": t,
                "vehicle_x": round(x, 2),
                "vehicle_y": round(y, 2),
                "vehicle_speed": round(max(0.0, sp), 2),
                "vehicle_angle": round(ang, 2),
                "vehicle_route": route,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep=";", index=False)
    n_veh = out_df["vehicle_id"].nunique()
    print(f"    → {output_path.name}: {len(out_df)} rows, {n_veh} vehicles")


def main():
    print(f"Base CSV: {BASE_CSV}")

    # Verify base CSV
    base = pd.read_csv(BASE_CSV, sep=";")
    print(f"Base: {len(base)} rows, {base['vehicle_id'].nunique()} vehicles, "
          f"t={base['data_timestep'].min():.0f}-{base['data_timestep'].max():.0f}")

    for speed_kmh, factor in sorted(SPEED_FACTORS.items()):
        generate_speed_csv(speed_kmh, factor)

    print("\nDone. All CSVs:")
    for p in sorted(SUMO_DIR.glob("unified_vehicle_trajectories_*kmh.csv")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
