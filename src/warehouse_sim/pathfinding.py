from __future__ import annotations

import heapq
from itertools import count
from time import perf_counter

from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlanStep, PlannerConfig, PlannerDiagnostics, Position
from warehouse_sim.reservation import ReservationTable


def plan_path_space_time(
    grid: GridMap,
    start: Position,
    goal: Position,
    start_time: int,
    max_time: int,
    planner: PlannerConfig,
    reservations: ReservationTable | None = None,
    robot_id: str | None = None,
    blocked_cells: set[Position] | None = None,
) -> tuple[list[PlanStep] | None, PlannerDiagnostics]:
    """Generic time-extended planner supporting multiple f-cost strategies."""

    t0 = perf_counter()
    blocked = blocked_cells or set()

    if not _is_traversable(grid, start, blocked) or not _is_traversable(grid, goal, blocked):
        return None, _diag(planner, 0, t0, path_cost=0, found=False)
    if start_time > max_time:
        return None, _diag(planner, 0, t0, path_cost=0, found=False)
    if reservations and reservations.is_vertex_reserved(start_time, start, robot_id):
        return None, _diag(planner, 0, t0, path_cost=0, found=False)

    start_state = (start, start_time)
    frontier: list[tuple[float, int, tuple[Position, int]]] = []
    tie = count()
    start_h = _heuristic(grid, start, goal, planner)
    heapq.heappush(frontier, (start_h, next(tie), start_state))

    came_from: dict[tuple[Position, int], tuple[Position, int] | None] = {start_state: None}
    cost_so_far: dict[tuple[Position, int], int] = {start_state: 0}
    expanded_nodes = 0

    while frontier:
        _, _, (current_pos, current_time) = heapq.heappop(frontier)
        expanded_nodes += 1

        if current_pos == goal:
            path = _reconstruct_path(came_from, (current_pos, current_time))
            return path, _diag(planner, expanded_nodes, t0, len(path) - 1, found=True)

        if current_time >= max_time:
            continue

        for nxt in _neighbors_with_wait_blocked(grid, current_pos, blocked):
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
            priority = float(new_cost + _heuristic(grid, nxt, goal, planner))
            heapq.heappush(frontier, (priority, next(tie), state))

    return None, _diag(planner, expanded_nodes, t0, path_cost=0, found=False)


def astar_space_time(
    grid: GridMap,
    start: Position,
    goal: Position,
    start_time: int,
    max_time: int,
    reservations: ReservationTable | None = None,
    robot_id: str | None = None,
) -> list[PlanStep] | None:
    """Backward-compatible wrapper for A* search."""

    path, _ = plan_path_space_time(
        grid=grid,
        start=start,
        goal=goal,
        start_time=start_time,
        max_time=max_time,
        planner=PlannerConfig(algorithm="astar", heuristic_weight=1.0),
        reservations=reservations,
        robot_id=robot_id,
    )
    return path


def _heuristic(
    grid: GridMap,
    current: Position,
    goal: Position,
    planner: PlannerConfig,
) -> float:
    if planner.algorithm == "dijkstra":
        return 0.0

    h = float(grid.manhattan(current, goal))
    if planner.algorithm == "weighted_astar":
        return float(max(1.0, planner.heuristic_weight) * h)
    return h


def _neighbors_with_wait_blocked(
    grid: GridMap,
    pos: Position,
    blocked_cells: set[Position],
) -> list[Position]:
    x, y = pos
    candidates = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    neighbors: list[Position] = []
    for cell in candidates:
        if not _is_traversable(grid, cell, blocked_cells):
            continue
        neighbors.append(cell)
    return neighbors


def _is_traversable(grid: GridMap, pos: Position, blocked_cells: set[Position]) -> bool:
    return grid.is_valid(pos) and pos not in blocked_cells


def _diag(
    planner: PlannerConfig,
    expanded_nodes: int,
    started_at: float,
    path_cost: int,
    found: bool,
) -> PlannerDiagnostics:
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    return PlannerDiagnostics(
        algorithm=planner.algorithm,
        expanded_nodes=expanded_nodes,
        planning_time_ms=round(elapsed_ms, 3),
        path_cost=path_cost,
        found_path=found,
    )


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
