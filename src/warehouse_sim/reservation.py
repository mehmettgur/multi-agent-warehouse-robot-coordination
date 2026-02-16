from __future__ import annotations

from collections import defaultdict

from warehouse_sim.models import PlanStep, Position


class ReservationTable:
    """Cell-time and edge-time reservations used by prioritized planning."""

    def __init__(self) -> None:
        self._vertex: dict[int, dict[Position, str]] = defaultdict(dict)
        self._edge: dict[int, dict[tuple[Position, Position], str]] = defaultdict(dict)

    def reserve_vertex(self, time: int, pos: Position, robot_id: str) -> None:
        owner = self._vertex[time].get(pos)
        if owner is not None and owner != robot_id:
            raise ValueError(
                f"Vertex ({pos}, t={time}) already reserved by {owner}, cannot reserve for {robot_id}"
            )
        self._vertex[time][pos] = robot_id

    def reserve_edge(
        self, time: int, from_pos: Position, to_pos: Position, robot_id: str
    ) -> None:
        owner = self._edge[time].get((from_pos, to_pos))
        if owner is not None and owner != robot_id:
            raise ValueError(
                f"Edge ({from_pos}->{to_pos}, t={time}) already reserved by {owner}, cannot reserve for {robot_id}"
            )
        self._edge[time][(from_pos, to_pos)] = robot_id

    def reserve_path(self, robot_id: str, path: list[PlanStep]) -> None:
        for step in path:
            self.reserve_vertex(step.time, step.position, robot_id)
        for current, nxt in zip(path, path[1:]):
            self.reserve_edge(current.time, current.position, nxt.position, robot_id)

    def is_vertex_reserved(
        self, time: int, pos: Position, robot_id: str | None = None
    ) -> bool:
        owner = self._vertex.get(time, {}).get(pos)
        if owner is None:
            return False
        if robot_id is None:
            return True
        return owner != robot_id

    def is_edge_reserved(
        self,
        time: int,
        from_pos: Position,
        to_pos: Position,
        robot_id: str | None = None,
    ) -> bool:
        owner = self._edge.get(time, {}).get((from_pos, to_pos))
        if owner is None:
            return False
        if robot_id is None:
            return True
        return owner != robot_id

    def snapshot_vertex(self, time: int) -> dict[Position, str]:
        return dict(self._vertex.get(time, {}))
