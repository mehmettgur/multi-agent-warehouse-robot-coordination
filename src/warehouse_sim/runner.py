from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from warehouse_sim.loader import load_scenario
from warehouse_sim.models import Mode
from warehouse_sim.simulator import run_simulation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    scenario_path: str,
    seed: int | None,
    output_dir: str,
) -> dict:
    config = load_scenario(scenario_path)
    use_seed = config.seed if seed is None else seed

    baseline = run_simulation(config=config, mode="baseline", seed=use_seed)
    coordinated = run_simulation(config=config, mode="coordinated", seed=use_seed)

    payload = {
        "scenario": config.name,
        "seed": use_seed,
        "baseline": baseline.metrics.to_dict(),
        "coordinated": coordinated.metrics.to_dict(),
    }

    out_dir = Path(output_dir)
    scenario_slug = Path(scenario_path).stem
    _write_json(out_dir / f"{scenario_slug}_comparison.json", payload)
    _write_csv(
        out_dir / f"{scenario_slug}_comparison.csv",
        [
            {"mode": "baseline", **baseline.metrics.to_dict()},
            {"mode": "coordinated", **coordinated.metrics.to_dict()},
        ],
    )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Warehouse multi-agent simulator runner")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to scenario JSON file",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "coordinated"],
        default="coordinated",
        help="Run mode for single execution",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override scenario seed")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and coordinated modes and output comparison",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON/CSV outputs",
    )

    args = parser.parse_args()

    if args.compare:
        payload = run_comparison(
            scenario_path=args.scenario,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(json.dumps(payload, indent=2))
        return

    config = load_scenario(args.scenario)
    mode: Mode = args.mode
    result = run_simulation(config=config, mode=mode, seed=args.seed)

    out_dir = Path(args.output_dir)
    scenario_slug = Path(args.scenario).stem
    _write_json(out_dir / f"{scenario_slug}_{mode}.json", result.to_dict())
    print(json.dumps(result.metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
