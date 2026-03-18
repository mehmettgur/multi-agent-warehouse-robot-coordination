from __future__ import annotations

import csv
from pathlib import Path

from warehouse_sim.paper_tables import generate_paper_tables
from warehouse_sim.runner import run_suite
from warehouse_sim.scenario_catalog import CORE_SCENARIOS


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_main_suite_outputs_expected_rows_and_latex(tmp_path: Path) -> None:
    payload = run_suite("main", str(tmp_path), with_latex=True)

    assert payload["num_rows"] == len(CORE_SCENARIOS) * 2
    assert "main" in payload["tables"]

    main_csv = next(Path(path) for path in payload["tables"]["main"] if path.endswith(".csv"))
    main_tex = next(Path(path) for path in payload["tables"]["main"] if path.endswith(".tex"))
    assert main_csv.exists()
    assert main_tex.exists()

    rows = _read_csv(main_csv)
    assert len(rows) == len(CORE_SCENARIOS) * 2
    assert {row["Senaryo"] for row in rows} == set(CORE_SCENARIOS)
    assert "narrow_corridor" not in {row["Senaryo"] for row in rows}


def test_planner_suite_contains_three_planners(tmp_path: Path) -> None:
    payload = run_suite("planner", str(tmp_path), with_latex=True)
    raw_rows = _read_csv(payload["raw_csv"])

    assert {row["planner"] for row in raw_rows} == {"astar", "dijkstra", "weighted_astar"}
    assert all(row["suite"] == "planner" for row in raw_rows)


def test_coordination_suite_contains_expected_variants(tmp_path: Path) -> None:
    payload = run_suite("coordination", str(tmp_path), with_latex=True)
    raw_rows = _read_csv(payload["raw_csv"])

    assert {row["coordination_variant"] for row in raw_rows} == {
        "independent",
        "priority_static",
        "vertex_only",
        "static_priority",
        "full",
    }
    assert all(row["suite"] == "coordination" for row in raw_rows)


def test_robustness_suite_aggregates_expected_scenarios(tmp_path: Path) -> None:
    payload = run_suite("robustness", str(tmp_path), with_latex=True)
    robustness_csv = next(
        Path(path) for path in payload["tables"]["robustness"] if path.endswith(".csv")
    )

    rows = _read_csv(robustness_csv)
    assert {row["Senaryo"] for row in rows} == {"dynamic_obstacle", "stochastic_delay"}
    assert len(rows) == 2


def test_all_suite_generates_figures_and_all_tables(tmp_path: Path) -> None:
    payload = run_suite("all", str(tmp_path), with_latex=True, with_figures=True)

    assert set(payload["tables"]) == {"main", "allocator", "planner", "coordination", "robustness"}
    assert {"swap_demo", "high_load_compare", "dynamic_obstacle", "manifest"} <= set(payload["figures"])

    for figure_group in payload["figures"].values():
        for path in figure_group:
            assert Path(path).exists()


def test_suite_csv_can_be_reexported_to_paper_tables(tmp_path: Path) -> None:
    payload = run_suite("main", str(tmp_path), with_latex=False)
    output_dir = tmp_path / "reexport"

    outputs = generate_paper_tables(
        input_csv=Path(payload["raw_csv"]),
        output_dir=output_dir,
        with_latex=True,
    )

    assert isinstance(outputs, dict)
    assert "main" in outputs
    for path in outputs["main"]:
        assert path.exists()
