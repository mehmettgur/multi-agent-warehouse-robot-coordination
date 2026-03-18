from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from warehouse_sim.figures import generate_paper_figures
from warehouse_sim.loader import load_scenario
from warehouse_sim.models import AllocationPolicy, PlannerAlgorithm, PlannerConfig
from warehouse_sim.paper_tables import generate_suite_tables
from warehouse_sim.scenario_catalog import (
    list_available_scenarios,
    scenario_path,
    scenario_paths,
    scenarios_for_suite,
)
from warehouse_sim.simulator import run_simulation

PaperSuite = Literal["main", "allocator", "planner", "robustness", "all"]
DEFAULT_ROBUSTNESS_SEEDS = [11, 17, 23, 31, 37]


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


def _planner_config(algorithm: PlannerAlgorithm, heuristic_weight: float = 1.0) -> PlannerConfig:
    return PlannerConfig(algorithm=algorithm, heuristic_weight=heuristic_weight)


def _run_row(
    suite: str,
    scenario_path_str: str,
    planner: PlannerConfig,
    allocator: AllocationPolicy,
    mode: str,
    seed: int,
) -> dict:
    config = load_scenario(scenario_path_str)
    result = run_simulation(
        config=config,
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        planner_override=planner,
        allocator_override=allocator,
    )
    return {
        "suite": suite,
        "scenario": config.name,
        "scenario_file": Path(scenario_path_str).name,
        "seed": seed,
        "mode": mode,
        "planner": planner.algorithm,
        "heuristic_weight": planner.heuristic_weight,
        "allocator": allocator,
        **result.metrics.to_dict(),
    }


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

    for scenario_path_str in scenario_paths:
        config = load_scenario(scenario_path_str)
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
                            "scenario_file": Path(scenario_path_str).name,
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


def _main_suite_rows() -> list[dict]:
    planner = _planner_config("astar", 1.0)
    rows: list[dict] = []
    for path_str in scenario_paths(scenarios_for_suite("main")):
        config = load_scenario(path_str)
        for mode in ["baseline", "coordinated"]:
            rows.append(
                _run_row(
                    suite="main",
                    scenario_path_str=path_str,
                    planner=planner,
                    allocator="greedy",
                    mode=mode,
                    seed=config.seed,
                )
            )
    return rows


def _allocator_suite_rows() -> list[dict]:
    planner = _planner_config("astar", 1.0)
    rows: list[dict] = []
    for path_str in scenario_paths(scenarios_for_suite("allocator")):
        config = load_scenario(path_str)
        for allocator in ["greedy", "hungarian"]:
            rows.append(
                _run_row(
                    suite="allocator",
                    scenario_path_str=path_str,
                    planner=planner,
                    allocator=allocator,
                    mode="coordinated",
                    seed=config.seed,
                )
            )
    return rows


def _planner_suite_rows() -> list[dict]:
    rows: list[dict] = []
    planners = [
        _planner_config("astar", 1.0),
        _planner_config("dijkstra", 1.0),
        _planner_config("weighted_astar", 1.4),
    ]
    for path_str in scenario_paths(scenarios_for_suite("planner")):
        config = load_scenario(path_str)
        for planner in planners:
            rows.append(
                _run_row(
                    suite="planner",
                    scenario_path_str=path_str,
                    planner=planner,
                    allocator="hungarian",
                    mode="coordinated",
                    seed=config.seed,
                )
            )
    return rows


def _robustness_suite_rows(seeds: list[int]) -> list[dict]:
    planner = _planner_config("astar", 1.0)
    rows: list[dict] = []

    dynamic_path = str(scenario_path("dynamic_obstacle"))
    dynamic_config = load_scenario(dynamic_path)
    rows.append(
        _run_row(
            suite="robustness",
            scenario_path_str=dynamic_path,
            planner=planner,
            allocator="hungarian",
            mode="coordinated",
            seed=dynamic_config.seed,
        )
    )

    stochastic_path = str(scenario_path("stochastic_delay"))
    for seed in seeds:
        rows.append(
            _run_row(
                suite="robustness",
                scenario_path_str=stochastic_path,
                planner=planner,
                allocator="hungarian",
                mode="coordinated",
                seed=seed,
            )
        )
    return rows


def _suite_rows(suite: PaperSuite, seeds: list[int]) -> list[dict]:
    if suite == "main":
        return _main_suite_rows()
    if suite == "allocator":
        return _allocator_suite_rows()
    if suite == "planner":
        return _planner_suite_rows()
    if suite == "robustness":
        return _robustness_suite_rows(seeds)
    if suite == "all":
        rows: list[dict] = []
        rows.extend(_main_suite_rows())
        rows.extend(_allocator_suite_rows())
        rows.extend(_planner_suite_rows())
        rows.extend(_robustness_suite_rows(seeds))
        return rows
    raise ValueError(f"Unsupported suite: {suite}")


def run_suite(
    suite: PaperSuite,
    output_dir: str,
    seeds: list[int] | None = None,
    with_latex: bool = False,
    with_figures: bool = False,
) -> dict:
    use_seeds = seeds or list(DEFAULT_ROBUSTNESS_SEEDS)
    rows = _suite_rows(suite, use_seeds)
    paper_dir = Path(output_dir) / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    raw_stem = "all_raw" if suite == "all" else f"{suite}_raw"
    raw_csv = paper_dir / f"{raw_stem}.csv"
    raw_json = paper_dir / f"{raw_stem}.json"
    _write_csv(raw_csv, rows)
    _write_json(
        raw_json,
        {
            "suite": suite,
            "seeds": use_seeds,
            "num_rows": len(rows),
            "rows": rows,
        },
    )

    table_outputs = generate_suite_tables(
        rows=rows,
        output_dir=paper_dir,
        with_latex=with_latex,
    )

    figure_outputs: dict[str, list[str]] = {}
    if with_figures:
        figures = generate_paper_figures(paper_dir)
        figure_outputs = {key: [str(path) for path in paths] for key, paths in figures.items()}

    payload = {
        "suite": suite,
        "seeds": use_seeds,
        "num_rows": len(rows),
        "raw_csv": str(raw_csv),
        "raw_json": str(raw_json),
        "tables": {
            key: [str(path) for path in paths] for key, paths in table_outputs.items()
        },
        "figures": figure_outputs,
    }
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
    return scenario_paths(list_available_scenarios(include_appendix=True, include_legacy=False))


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
        "--suite",
        choices=["main", "allocator", "planner", "robustness", "all"],
        default=None,
        help="Run canonical paper benchmark suite",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Seed list for robustness suite",
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
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Paper suite çıktıları için .tex tablolarını da üret",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Paper suite ile birlikte reproducible figür paketini üret",
    )

    args = parser.parse_args()

    planner = _parse_planner(args.planner, args.heuristic_weight)
    allocator: AllocationPolicy | None = args.allocator

    if args.suite is not None:
        payload = run_suite(
            suite=args.suite,
            output_dir=args.output_dir,
            seeds=args.seeds,
            with_latex=args.latex,
            with_figures=args.figures,
        )
        print(json.dumps(payload, indent=2))
        return

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
        raise SystemExit("--scenario is required unless --ablation or --suite is used")

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
    run_seed = config.seed if args.seed is None else args.seed
    result = run_simulation(
        config=config,
        mode=args.mode,
        seed=run_seed,
        planner_override=planner,
        allocator_override=allocator,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
