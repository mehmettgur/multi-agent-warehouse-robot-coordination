from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlannerConfig
from warehouse_sim.pathfinding import plan_path_space_time
from warehouse_sim.reservation import ReservationTable


def _path_cost(path):
    if path is None:
        return None
    return len(path) - 1


def test_astar_finds_shortest_path_with_manhattan_heuristic() -> None:
    grid = GridMap(width=5, height=5, obstacles={(2, 2)})
    path, diag = plan_path_space_time(
        grid=grid,
        start=(0, 0),
        goal=(4, 0),
        start_time=0,
        max_time=20,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
    )

    assert path is not None
    assert path[0].position == (0, 0)
    assert path[-1].position == (4, 0)
    assert _path_cost(path) == 4
    assert diag.found_path


def test_dijkstra_matches_astar_path_cost() -> None:
    grid = GridMap(width=7, height=5, obstacles={(3, 1), (3, 2), (3, 3)})

    astar_path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 2),
        goal=(6, 2),
        start_time=0,
        max_time=50,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
    )
    dijkstra_path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 2),
        goal=(6, 2),
        start_time=0,
        max_time=50,
        planner=PlannerConfig(algorithm="dijkstra", heuristic_weight=1.0),
    )

    assert _path_cost(astar_path) == _path_cost(dijkstra_path)


def test_weighted_astar_cost_is_not_better_than_astar() -> None:
    grid = GridMap(width=9, height=7, obstacles={(4, y) for y in range(1, 6) if y != 3})

    astar_path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 3),
        goal=(8, 3),
        start_time=0,
        max_time=70,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
    )
    weighted_path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 3),
        goal=(8, 3),
        start_time=0,
        max_time=70,
        planner=PlannerConfig(algorithm="weighted_astar", heuristic_weight=2.0),
    )

    assert astar_path is not None and weighted_path is not None
    assert _path_cost(weighted_path) >= _path_cost(astar_path)


def test_astar_uses_wait_when_next_cell_is_temporarily_reserved() -> None:
    grid = GridMap(width=3, height=1, obstacles=set())
    reservations = ReservationTable()
    reservations.reserve_vertex(time=1, pos=(1, 0), robot_id="R1")

    path, _ = plan_path_space_time(
        grid=grid,
        start=(0, 0),
        goal=(2, 0),
        start_time=0,
        max_time=5,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
        reservations=reservations,
        robot_id="R2",
    )

    assert path is not None
    positions = [step.position for step in path]
    assert positions[:4] == [(0, 0), (0, 0), (1, 0), (2, 0)]
