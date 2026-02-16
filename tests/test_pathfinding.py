from warehouse_sim.grid import GridMap
from warehouse_sim.pathfinding import astar_space_time
from warehouse_sim.reservation import ReservationTable


def test_astar_finds_shortest_path_with_manhattan_heuristic() -> None:
    grid = GridMap(width=5, height=5, obstacles={(2, 2)})
    path = astar_space_time(
        grid=grid,
        start=(0, 0),
        goal=(4, 0),
        start_time=0,
        max_time=20,
    )

    assert path is not None
    positions = [step.position for step in path]
    assert positions[0] == (0, 0)
    assert positions[-1] == (4, 0)
    assert len(path) - 1 == 4


def test_astar_uses_wait_when_next_cell_is_temporarily_reserved() -> None:
    grid = GridMap(width=3, height=1, obstacles=set())
    reservations = ReservationTable()
    reservations.reserve_vertex(time=1, pos=(1, 0), robot_id="R1")

    path = astar_space_time(
        grid=grid,
        start=(0, 0),
        goal=(2, 0),
        start_time=0,
        max_time=5,
        reservations=reservations,
        robot_id="R2",
    )

    assert path is not None
    positions = [step.position for step in path]
    assert positions[:4] == [(0, 0), (0, 0), (1, 0), (2, 0)]
