import random

from warehouse_sim.events import EventEngine
from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlannerConfig
from warehouse_sim.pathfinding import plan_path_space_time


def test_temp_block_event_marks_cell_as_blocked() -> None:
    engine = EventEngine(
        events=[
            {"type": "temp_block", "cell": [2, 2], "start_tick": 3, "end_tick": 5},
        ]
    )

    assert (2, 2) not in engine.active_blocked_cells(2)
    assert (2, 2) in engine.active_blocked_cells(3)
    assert (2, 2) in engine.active_blocked_cells(5)
    assert (2, 2) not in engine.active_blocked_cells(6)


def test_stochastic_delay_is_deterministic_with_same_seed() -> None:
    events = [
        {"type": "stochastic_delay", "probability": 0.4, "start_tick": 1, "end_tick": 3}
    ]
    engine = EventEngine(events=events)

    rng_a = random.Random(123)
    rng_b = random.Random(123)

    delayed_a, cnt_a = engine.delayed_robots(2, ["R1", "R2", "R3"], rng_a)
    delayed_b, cnt_b = engine.delayed_robots(2, ["R1", "R2", "R3"], rng_b)

    assert delayed_a == delayed_b
    assert cnt_a == cnt_b


def test_dynamic_blocked_cell_is_avoided_by_planner() -> None:
    grid = GridMap(width=5, height=1, obstacles=set())
    path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 0),
        goal=(4, 0),
        start_time=0,
        max_time=10,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
        blocked_cells={(1, 0)},
    )

    assert path is None
