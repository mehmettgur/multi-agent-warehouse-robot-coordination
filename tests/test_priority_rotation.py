from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.models import RobotState, TaskState


def _build_conflicting_line_case() -> tuple[TrafficManagerAgent, dict[str, RobotState], dict[str, TaskState]]:
    grid = GridMap(width=5, height=1, obstacles=set())
    manager = TrafficManagerAgent(grid=grid, max_ticks=20)
    robots = {
        "R1": RobotState(
            robot_id="R1",
            position=(1, 0),
            current_task_id="T1",
            phase="to_dropoff",
        ),
        "R2": RobotState(
            robot_id="R2",
            position=(3, 0),
            current_task_id="T2",
            phase="to_dropoff",
        ),
    }
    tasks = {
        "T1": TaskState(
            task_id="T1",
            pickup=(0, 0),
            dropoff=(4, 0),
            release_tick=0,
            assigned_robot_id="R1",
            pickup_tick=0,
        ),
        "T2": TaskState(
            task_id="T2",
            pickup=(4, 0),
            dropoff=(0, 0),
            release_tick=0,
            assigned_robot_id="R2",
            pickup_tick=0,
        ),
    }
    return manager, robots, tasks


def test_priority_rotation_changes_first_mover_by_tick() -> None:
    manager, robots, tasks = _build_conflicting_line_case()

    tick0_paths, _ = manager.plan_and_reserve(
        robots=robots,
        tasks=tasks,
        tick=0,
        wait_streaks={"R1": 0, "R2": 0},
    )
    tick1_paths, _ = manager.plan_and_reserve(
        robots=robots,
        tasks=tasks,
        tick=1,
        wait_streaks={"R1": 0, "R2": 0},
    )

    assert list(tick0_paths.keys())[0] == "R1"
    assert tick0_paths["R1"][1].position == (2, 0)
    assert list(tick1_paths.keys())[0] == "R2"
    assert tick1_paths["R2"][1].position == (2, 0)


def test_wait_streak_overrides_eta_tie_for_deadlock_break() -> None:
    manager, robots, tasks = _build_conflicting_line_case()

    paths, _ = manager.plan_and_reserve(
        robots=robots,
        tasks=tasks,
        tick=0,
        wait_streaks={"R1": 0, "R2": 5},
    )

    assert list(paths.keys())[0] == "R2"
    assert paths["R2"][1].position == (2, 0)
