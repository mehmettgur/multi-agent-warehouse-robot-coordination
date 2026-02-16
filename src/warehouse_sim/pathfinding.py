from __future__ import annotations

import heapq
from itertools import count

from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlanStep, Position
from warehouse_sim.reservation import ReservationTable


def astar_space_time(
    grid: GridMap,
    start: Position,
    goal: Position,
    start_time: int,
    max_time: int,
    reservations: ReservationTable | None = None,
    robot_id: str | None = None,
) -> list[PlanStep] | None:
    """A* in time-extended space with WAIT action and optional reservations."""

    if not grid.is_valid(start) or not grid.is_valid(goal):
        return None
    if start_time > max_time:
        return None
    if reservations and reservations.is_vertex_reserved(start_time, start, robot_id):
        return None

    start_state = (start, start_time)
    frontier: list[tuple[int, int, tuple[Position, int]]] = []
    tie = count()
    start_h = grid.manhattan(start, goal)
    heapq.heappush(frontier, (start_h, next(tie), start_state))

    came_from: dict[tuple[Position, int], tuple[Position, int] | None] = {start_state: None}
    cost_so_far: dict[tuple[Position, int], int] = {start_state: 0}

    while frontier:
        _, _, (current_pos, current_time) = heapq.heappop(frontier)

        if current_pos == goal:
            return _reconstruct_path(came_from, (current_pos, current_time))

        if current_time >= max_time:
            continue

        for nxt in grid.neighbors_with_wait(current_pos):
            next_time = current_time + 1
            if next_time > max_time:
                continue

            if reservations and reservations.is_vertex_reserved(next_time, nxt, robot_id):
                continue
            if reservations and reservations.is_edge_reserved(
                current_time, nxt, current_pos, robot_id
            ):
                continue

            state = (nxt, next_time)
            new_cost = cost_so_far[(current_pos, current_time)] + 1
            if new_cost >= cost_so_far.get(state, 10**12):
                continue

            cost_so_far[state] = new_cost
            came_from[state] = (current_pos, current_time)
            priority = new_cost + grid.manhattan(nxt, goal)
            heapq.heappush(frontier, (priority, next(tie), state))

    return None


def _reconstruct_path(
    came_from: dict[tuple[Position, int], tuple[Position, int] | None],
    terminal: tuple[Position, int],
) -> list[PlanStep]:
    states: list[tuple[Position, int]] = []
    cursor: tuple[Position, int] | None = terminal
    while cursor is not None:
        states.append(cursor)
        cursor = came_from[cursor]
    states.reverse()
    return [PlanStep(position=pos, time=t) for pos, t in states]
