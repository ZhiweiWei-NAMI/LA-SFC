from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MASAC runtime_debug.jsonl heartbeat traces.")
    parser.add_argument("--root", type=Path, default=Path("experiment_artifacts/raw_data"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    roots = debug_files(args.root)
    if not roots:
        print(f"No runtime_debug.jsonl files found under {args.root}")
        return 1
    now = time.time()
    for path in roots:
        rows = read_jsonl(path)
        if not rows:
            continue
        print(f"\n{path}")
        print(last_event_line(rows[-1], now))
        for line in slow_transition_lines(rows, max(1, int(args.top))):
            print(line)
    return 0


def debug_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("**/runtime_debug.jsonl"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    except OSError:
        return []
    return rows


def last_event_line(row: Mapping[str, Any], now: float) -> str:
    timestamp = as_float(row.get("time_s"))
    age = int(max(0.0, now - timestamp)) if timestamp is not None else -1
    where = location(row)
    return f"last={row.get('event', '-')} {where} age_s={age} pid={row.get('pid', '-')}"


def slow_transition_lines(rows: list[dict[str, Any]], top: int) -> Iterable[str]:
    transitions: list[tuple[float, str, str, dict[str, Any]]] = []
    for prev, current in zip(rows, rows[1:]):
        start = as_float(prev.get("time_s"))
        end = as_float(current.get("time_s"))
        if start is None or end is None:
            continue
        delta = max(0.0, end - start)
        transitions.append((delta, str(prev.get("event", "-")), str(current.get("event", "-")), current))
    for delta, prev_event, current_event, row in sorted(transitions, key=lambda item: item[0], reverse=True)[:top]:
        print_row = {
            "episode": row.get("episode", "-"),
            "step": row.get("step", "-"),
            "update": row.get("update_index", "-"),
            "scenario": row.get("scenario", "-"),
        }
        details = " ".join(f"{key}={value}" for key, value in print_row.items())
        yield f"slow {delta:.3f}s {prev_event}->{current_event} {details}"


def location(row: Mapping[str, Any]) -> str:
    parts = []
    for key in ("baseline", "seed", "episode", "step", "update_index", "scenario"):
        if key in row:
            parts.append(f"{key}={row[key]}")
    return " ".join(parts)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
