from warehouse_sim.models import PlanStep
from warehouse_sim.reservation import ReservationTable


def test_vertex_reservation_conflict_checks() -> None:
    table = ReservationTable()
    table.reserve_vertex(time=2, pos=(1, 1), robot_id="R1")

    assert table.is_vertex_reserved(time=2, pos=(1, 1))
    assert table.is_vertex_reserved(time=2, pos=(1, 1), robot_id="R2")
    assert not table.is_vertex_reserved(time=2, pos=(1, 1), robot_id="R1")


def test_edge_reservation_supports_swap_collision_check_pattern() -> None:
    table = ReservationTable()
    table.reserve_edge(time=3, from_pos=(1, 1), to_pos=(1, 2), robot_id="R1")

    assert table.is_edge_reserved(time=3, from_pos=(1, 1), to_pos=(1, 2))
    assert not table.is_edge_reserved(time=3, from_pos=(1, 1), to_pos=(1, 2), robot_id="R1")
    assert table.is_edge_reserved(time=3, from_pos=(1, 1), to_pos=(1, 2), robot_id="R2")


def test_reserve_path_creates_vertex_and_edge_entries() -> None:
    table = ReservationTable()
    path = [
        PlanStep(position=(0, 0), time=0),
        PlanStep(position=(1, 0), time=1),
        PlanStep(position=(1, 1), time=2),
    ]

    table.reserve_path(robot_id="R1", path=path)

    assert table.is_vertex_reserved(time=1, pos=(1, 0))
    assert table.is_edge_reserved(time=0, from_pos=(0, 0), to_pos=(1, 0))
    assert table.is_edge_reserved(time=1, from_pos=(1, 0), to_pos=(1, 1))
