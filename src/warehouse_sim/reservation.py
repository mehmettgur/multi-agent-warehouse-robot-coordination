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
        reverse_owner = self._edge[time].get((to_pos, from_pos))
        if (
            reverse_owner is not None
            and reverse_owner != robot_id
            and from_pos != to_pos
        ):
            raise ValueError(
                f"Reverse edge ({to_pos}->{from_pos}, t={time}) already reserved by {reverse_owner}, cannot reserve for {robot_id}"
            )
        self._edge[time][(from_pos, to_pos)] = robot_id

    def reserve_path(self, robot_id: str, path: list[PlanStep]) -> None:
        # Pre-check entire path first to keep path reservation atomic.
        for step in path:
            owner = self._vertex.get(step.time, {}).get(step.position)
            if owner is not None and owner != robot_id:
                raise ValueError(
                    f"Vertex ({step.position}, t={step.time}) already reserved by {owner}, cannot reserve for {robot_id}"
                )

        for current, nxt in zip(path, path[1:]):
            owner = self._edge.get(current.time, {}).get((current.position, nxt.position))
            if owner is not None and owner != robot_id:
                raise ValueError(
                    f"Edge ({current.position}->{nxt.position}, t={current.time}) already reserved by {owner}, cannot reserve for {robot_id}"
                )
            reverse_owner = self._edge.get(current.time, {}).get((nxt.position, current.position))
            if (
                reverse_owner is not None
                and reverse_owner != robot_id
                and current.position != nxt.position
            ):
                raise ValueError(
                    f"Reverse edge ({nxt.position}->{current.position}, t={current.time}) already reserved by {reverse_owner}, cannot reserve for {robot_id}"
                )

        for step in path:
            self._vertex[step.time][step.position] = robot_id
        for current, nxt in zip(path, path[1:]):
            self._edge[current.time][(current.position, nxt.position)] = robot_id

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
