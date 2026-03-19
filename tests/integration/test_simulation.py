from pathlib import Path

from warehouse_sim.loader import load_scenario
from warehouse_sim.models import CoordinationConfig, PlannerConfig
from warehouse_sim.runner import run_ablation, run_comparison
from warehouse_sim.simulator import run_simulation


def test_coordinated_has_zero_collisions_in_narrow_corridor_swap() -> None:
    config = load_scenario("configs/scenarios/narrow_corridor_swap.json")
    result = run_simulation(config=config, mode="coordinated", seed=17)
    assert result.metrics.collision_count == 0


def test_baseline_has_conflicts_in_narrow_corridor_swap() -> None:
    config = load_scenario("configs/scenarios/narrow_corridor_swap.json")
    result = run_simulation(config=config, mode="baseline", seed=17)
    assert result.metrics.collision_count > 0


def test_priority_baseline_improves_over_independent_baseline_in_narrow_corridor_swap() -> None:
    config = load_scenario("configs/scenarios/narrow_corridor_swap.json")
    baseline = run_simulation(config=config, mode="baseline", seed=17)
    priority_baseline = run_simulation(
        config=config,
        mode="baseline_priority",
        seed=17,
        coordination_override=CoordinationConfig(
            variant="priority_static",
            edge_conflicts=True,
            dynamic_priority=False,
            micro_replan=False,
        ),
    )

    assert priority_baseline.metrics.completed_tasks >= baseline.metrics.completed_tasks
    assert priority_baseline.metrics.collision_count <= baseline.metrics.collision_count


def test_stochastic_delay_scenario_is_deterministic_with_same_seed() -> None:
    config = load_scenario("configs/scenarios/stochastic_delay.json")
    result_a = run_simulation(config=config, mode="coordinated", seed=123)
    result_b = run_simulation(config=config, mode="coordinated", seed=123)

    metrics_a = result_a.metrics.to_dict()
    metrics_b = result_b.metrics.to_dict()
    metrics_a.pop("planner_time_ms_total", None)
    metrics_b.pop("planner_time_ms_total", None)

    assert metrics_a == metrics_b
    assert [snap.to_dict() for snap in result_a.timeline] == [
        snap.to_dict() for snap in result_b.timeline
    ]


def test_dynamic_obstacle_scenario_runs_with_event_metrics() -> None:
    config = load_scenario("configs/scenarios/dynamic_obstacle.json")
    result = run_simulation(config=config, mode="coordinated", seed=31)

    assert result.metrics.dynamic_block_replans >= 1


def test_merge_burst_priority_requires_dynamic_priority() -> None:
    config = load_scenario("configs/scenarios/merge_burst_priority.json")
    static_priority = run_simulation(
        config=config,
        mode="coordinated",
        seed=41,
        coordination_override=CoordinationConfig(
            variant="static_priority",
            edge_conflicts=True,
            dynamic_priority=False,
            micro_replan=True,
        ),
    )
    full = run_simulation(
        config=config,
        mode="coordinated",
        seed=41,
        coordination_override=CoordinationConfig(
            variant="full",
            edge_conflicts=True,
            dynamic_priority=True,
            micro_replan=True,
        ),
    )

    assert full.metrics.completed_tasks > static_priority.metrics.completed_tasks
    assert full.metrics.wait_count < static_priority.metrics.wait_count


def test_parking_bay_micro_replan_beats_no_micro_replan() -> None:
    config = load_scenario("configs/scenarios/parking_bay_micro_replan.json")
    no_micro = run_simulation(
        config=config,
        mode="coordinated",
        seed=13,
        coordination_override=CoordinationConfig(
            variant="no_micro_replan",
            edge_conflicts=True,
            dynamic_priority=True,
            micro_replan=False,
        ),
    )
    full = run_simulation(
        config=config,
        mode="coordinated",
        seed=13,
        coordination_override=CoordinationConfig(
            variant="full",
            edge_conflicts=True,
            dynamic_priority=True,
            micro_replan=True,
        ),
    )

    assert full.metrics.collision_count < no_micro.metrics.collision_count
    assert full.metrics.makespan < no_micro.metrics.makespan
    assert full.metrics.micro_replanning_count > 0


def test_comparison_runner_returns_both_modes_with_planner_override(tmp_path: Path) -> None:
    payload = run_comparison(
        scenario_path="configs/scenarios/narrow_corridor_swap.json",
        seed=17,
        output_dir=str(tmp_path),
        planner=PlannerConfig(algorithm="dijkstra", heuristic_weight=1.0),
        allocator="hungarian",
    )
    assert "baseline" in payload
    assert "coordinated" in payload
    assert payload["planner"] == "dijkstra"


def test_ablation_runner_outputs_expected_row_count(tmp_path: Path) -> None:
    payload = run_ablation(
        scenario_paths=[
            "configs/scenarios/narrow_corridor_swap.json",
            "configs/scenarios/intersection_4way_crossing.json",
        ],
        output_dir=str(tmp_path),
        planners=[
            PlannerConfig(algorithm="astar", heuristic_weight=1.0),
            PlannerConfig(algorithm="dijkstra", heuristic_weight=1.0),
        ],
        allocators=["greedy"],
        seed_override=7,
    )

    assert payload["num_rows"] == 8
    csv_path = Path(payload["csv"])
    json_path = Path(payload["json"])
    assert csv_path.exists()
    assert json_path.exists()
