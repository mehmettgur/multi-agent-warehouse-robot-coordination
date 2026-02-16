from warehouse_sim.loader import load_scenario
from warehouse_sim.runner import run_comparison
from warehouse_sim.simulator import run_simulation


def test_coordinated_has_zero_collisions_in_narrow_corridor() -> None:
    config = load_scenario("configs/scenarios/narrow_corridor.json")
    result = run_simulation(config=config, mode="coordinated", seed=7)
    assert result.metrics.collision_count == 0


def test_baseline_has_conflicts_in_narrow_corridor() -> None:
    config = load_scenario("configs/scenarios/narrow_corridor.json")
    result = run_simulation(config=config, mode="baseline", seed=7)
    assert result.metrics.collision_count > 0


def test_coordinated_dense_scenario_completes_all_tasks() -> None:
    config = load_scenario("configs/scenarios/dense_tasks.json")
    result = run_simulation(config=config, mode="coordinated", seed=21)
    assert result.metrics.completed_tasks == result.metrics.total_tasks


def test_deterministic_results_with_same_seed() -> None:
    config = load_scenario("configs/scenarios/dense_tasks.json")
    result_a = run_simulation(config=config, mode="coordinated", seed=99)
    result_b = run_simulation(config=config, mode="coordinated", seed=99)

    assert result_a.metrics.to_dict() == result_b.metrics.to_dict()
    assert [snap.to_dict() for snap in result_a.timeline] == [
        snap.to_dict() for snap in result_b.timeline
    ]


def test_comparison_runner_returns_both_modes() -> None:
    payload = run_comparison(
        scenario_path="configs/scenarios/narrow_corridor.json",
        seed=7,
        output_dir="results",
    )
    assert "baseline" in payload
    assert "coordinated" in payload
