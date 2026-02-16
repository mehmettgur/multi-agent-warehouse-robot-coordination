from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from warehouse_sim.loader import load_scenario
from warehouse_sim.models import AllocationPolicy, Mode, PlannerAlgorithm, PlannerConfig
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
    planner: PlannerConfig | None = None,
    allocator: AllocationPolicy | None = None,
) -> dict:
    config = load_scenario(scenario_path)
    use_seed = config.seed if seed is None else seed
    baseline_allocator = allocator if allocator is not None else "greedy"
    coordinated_allocator = allocator if allocator is not None else config.allocator_policy

    baseline = run_simulation(
        config=config,
        mode="baseline",
        seed=use_seed,
        planner_override=planner,
        allocator_override=allocator,
    )
    coordinated = run_simulation(
        config=config,
        mode="coordinated",
        seed=use_seed,
        planner_override=planner,
        allocator_override=allocator,
    )

    payload = {
        "scenario": config.name,
        "seed": use_seed,
        "planner": (planner.algorithm if planner else config.planner.algorithm),
        "heuristic_weight": (
            planner.heuristic_weight if planner else config.planner.heuristic_weight
        ),
        "baseline_allocator": baseline_allocator,
        "coordinated_allocator": coordinated_allocator,
        "baseline": baseline.metrics.to_dict(),
        "coordinated": coordinated.metrics.to_dict(),
    }

    out_dir = Path(output_dir)
    scenario_slug = Path(scenario_path).stem
    suffix = payload["planner"]
    _write_json(out_dir / f"{scenario_slug}_{suffix}_comparison.json", payload)
    _write_csv(
        out_dir / f"{scenario_slug}_{suffix}_comparison.csv",
        [
            {
                "mode": "baseline",
                "planner": payload["planner"],
                "allocator": baseline_allocator,
                **baseline.metrics.to_dict(),
            },
            {
                "mode": "coordinated",
                "planner": payload["planner"],
                "allocator": coordinated_allocator,
                **coordinated.metrics.to_dict(),
            },
        ],
    )

    return payload


def run_ablation(
    scenario_paths: list[str],
    output_dir: str,
    planners: list[PlannerConfig],
    allocators: list[AllocationPolicy],
    seed_override: int | None,
) -> dict:
    rows: list[dict] = []

    for scenario_path in scenario_paths:
        config = load_scenario(scenario_path)
        run_seed = config.seed if seed_override is None else seed_override

        for planner in planners:
            for allocator in allocators:
                for mode in ["baseline", "coordinated"]:
                    result = run_simulation(
                        config=config,
                        mode=mode,
                        seed=run_seed,
                        planner_override=planner,
                        allocator_override=allocator,
                    )
                    rows.append(
                        {
                            "scenario": config.name,
                            "scenario_file": Path(scenario_path).name,
                            "seed": run_seed,
                            "mode": mode,
                            "planner": planner.algorithm,
                            "heuristic_weight": planner.heuristic_weight,
                            "allocator": allocator,
                            **result.metrics.to_dict(),
                        }
                    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    csv_out = out_dir / f"ablation_{timestamp}.csv"
    json_out = out_dir / f"ablation_{timestamp}.json"

    _write_csv(csv_out, rows)
    payload = {
        "generated_at_utc": timestamp,
        "rows": rows,
        "num_rows": len(rows),
        "csv": str(csv_out),
        "json": str(json_out),
    }
    _write_json(json_out, payload)
    return payload


def _parse_planner(
    planner_name: PlannerAlgorithm | None,
    heuristic_weight: float | None,
) -> PlannerConfig | None:
    if planner_name is None:
        return None
    if planner_name == "weighted_astar":
        weight = 1.4 if heuristic_weight is None else heuristic_weight
    else:
        weight = 1.0 if heuristic_weight is None else heuristic_weight
    return PlannerConfig(algorithm=planner_name, heuristic_weight=weight)


def _default_scenarios() -> list[str]:
    return [str(path) for path in sorted(Path("configs/scenarios").glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Warehouse multi-agent simulator runner")
    parser.add_argument("--scenario", help="Path to scenario JSON file")
    parser.add_argument("--scenarios", nargs="*", help="Scenario list for ablation mode")
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
        "--ablation",
        action="store_true",
        help="Run ablation matrix across planners/allocators/scenarios",
    )
    parser.add_argument(
        "--planner",
        choices=["astar", "dijkstra", "weighted_astar"],
        default=None,
        help="Planner algorithm override",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=None,
        help="Heuristic weight for weighted A*",
    )
    parser.add_argument(
        "--allocator",
        choices=["greedy", "hungarian"],
        default=None,
        help="Allocator policy override",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON/CSV outputs",
    )

    args = parser.parse_args()

    planner = _parse_planner(args.planner, args.heuristic_weight)
    allocator: AllocationPolicy | None = args.allocator

    if args.ablation:
        scenarios = args.scenarios or ([args.scenario] if args.scenario else _default_scenarios())
        planners = [
            planner,
        ] if planner is not None else [
            PlannerConfig(algorithm="astar", heuristic_weight=1.0),
            PlannerConfig(algorithm="dijkstra", heuristic_weight=1.0),
            PlannerConfig(algorithm="weighted_astar", heuristic_weight=1.4),
        ]
        allocators = [allocator] if allocator is not None else ["greedy", "hungarian"]

        payload = run_ablation(
            scenario_paths=scenarios,
            output_dir=args.output_dir,
            planners=planners,
            allocators=allocators,
            seed_override=args.seed,
        )
        print(json.dumps(payload, indent=2))
        return

    if not args.scenario:
        raise SystemExit("--scenario is required unless --ablation is used with scenario defaults")

    if args.compare:
        payload = run_comparison(
            scenario_path=args.scenario,
            seed=args.seed,
            output_dir=args.output_dir,
            planner=planner,
            allocator=allocator,
        )
        print(json.dumps(payload, indent=2))
        return

    config = load_scenario(args.scenario)
    mode: Mode = args.mode
    result = run_simulation(
        config=config,
        mode=mode,
        seed=args.seed,
        planner_override=planner,
        allocator_override=allocator,
    )

    out_dir = Path(args.output_dir)
    scenario_slug = Path(args.scenario).stem
    planner_suffix = result.planner_algorithm
    _write_json(out_dir / f"{scenario_slug}_{mode}_{planner_suffix}.json", result.to_dict())
    print(json.dumps(result.metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
