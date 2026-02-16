from __future__ import annotations

from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlanStep, PlannerConfig, PlannerDiagnostics, RobotState, TaskState
from warehouse_sim.pathfinding import plan_path_space_time
from warehouse_sim.reservation import ReservationTable


class TrafficManagerAgent:
    """Prioritized planner with cell-time and edge-time reservations."""

    def __init__(self, grid: GridMap, max_ticks: int, planner: PlannerConfig) -> None:
        self.grid = grid
        self.max_ticks = max_ticks
        self.planner = planner

    def plan_and_reserve(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        tick: int,
        wait_streaks: dict[str, int] | None = None,
        blocked_cells: set[tuple[int, int]] | None = None,
        planner: PlannerConfig | None = None,
    ) -> tuple[dict[str, list[PlanStep]], ReservationTable, list[PlannerDiagnostics]]:
        reservation_table = ReservationTable()
        planned_paths: dict[str, list[PlanStep]] = {}
        diagnostics: list[PlannerDiagnostics] = []

        streaks = wait_streaks or {}
        planner_cfg = planner or self.planner
        blocked = blocked_cells or set()

        robot_order = sorted(robots)
        rank_by_robot = {robot_id: idx for idx, robot_id in enumerate(robot_order)}
        targets = {
            robot_id: self._target_for_robot(robot, tasks)
            for robot_id, robot in robots.items()
        }
        blocking_scores = {
            robot_id: sum(
                1
                for other_id, target in targets.items()
                if (
                    other_id != robot_id
                    and target is not None
                    and GridMap.manhattan(target, robots[robot_id].position) <= 1
                )
            )
            for robot_id in robots
        }
        active_targets = {target for target in targets.values() if target is not None}

        ordered_robot_ids = sorted(
            robots,
            key=lambda rid: self._priority_key(
                robot=robots[rid],
                tasks=tasks,
                blocking_score=blocking_scores.get(rid, 0),
                wait_streak=streaks.get(rid, 0),
                tick=tick,
                total_robots=len(robots),
                rank=rank_by_robot[rid],
            ),
        )

        for robot_id in ordered_robot_ids:
            robot = robots[robot_id]
            target = self._target_for_robot(robot, tasks)
            start = robot.position
            goal = target if target is not None else start

            if target is None and blocking_scores.get(robot_id, 0) > 0:
                preferred_next = self._preferred_yield_step(
                    start=start,
                    active_targets=active_targets,
                    blocked_cells=blocked,
                )
                next_pos = self._select_safe_next_step(
                    start=start,
                    target=preferred_next,
                    preferred_next=preferred_next,
                    reservation_table=reservation_table,
                    tick=tick,
                    blocked_cells=blocked,
                )
                path = [
                    PlanStep(position=start, time=tick),
                    PlanStep(position=next_pos, time=tick + 1),
                ]
                diagnostics.append(
                    PlannerDiagnostics(
                        algorithm=planner_cfg.algorithm,
                        expanded_nodes=0,
                        planning_time_ms=0.0,
                        path_cost=0 if next_pos == start else 1,
                        found_path=True,
                    )
                )
            else:
                max_time = min(self.max_ticks, tick + self.grid.width * self.grid.height * 2)
                path_hint, diag = plan_path_space_time(
                    grid=self.grid,
                    start=start,
                    goal=goal,
                    start_time=tick,
                    max_time=max_time,
                    planner=planner_cfg,
                    reservations=reservation_table,
                    robot_id=robot_id,
                    blocked_cells=blocked,
                )
                diagnostics.append(diag)

                if path_hint is None or len(path_hint) <= 1:
                    next_pos = self._select_safe_next_step(
                        start=start,
                        target=goal,
                        preferred_next=start,
                        reservation_table=reservation_table,
                        tick=tick,
                        blocked_cells=blocked,
                    )
                    path = [
                        PlanStep(position=start, time=tick),
                        PlanStep(position=next_pos, time=tick + 1),
                    ]
                elif self._can_reserve_path(path_hint, reservation_table, robot_id):
                    path = path_hint
                else:
                    next_pos = self._select_safe_next_step(
                        start=start,
                        target=goal,
                        preferred_next=path_hint[1].position,
                        reservation_table=reservation_table,
                        tick=tick,
                        blocked_cells=blocked,
                    )
                    path = [
                        PlanStep(position=start, time=tick),
                        PlanStep(position=next_pos, time=tick + 1),
                    ]

            if self._can_reserve_path(path, reservation_table, robot_id):
                reservation_table.reserve_path(robot_id, path)
            else:
                next_pos = self._select_safe_next_step(
                    start=start,
                    target=goal,
                    preferred_next=start,
                    reservation_table=reservation_table,
                    tick=tick,
                    blocked_cells=blocked,
                )
                fallback_path = [
                    PlanStep(position=start, time=tick),
                    PlanStep(position=next_pos, time=tick + 1),
                ]
                if self._can_reserve_path(fallback_path, reservation_table, robot_id):
                    reservation_table.reserve_path(robot_id, fallback_path)
                    path = fallback_path
                else:
                    path = [PlanStep(position=start, time=tick)]
                    if not reservation_table.is_vertex_reserved(tick, start, robot_id):
                        reservation_table.reserve_vertex(tick, start, robot_id)

            planned_paths[robot_id] = path

        return planned_paths, reservation_table, diagnostics

    def _preferred_yield_step(
        self,
        start: tuple[int, int],
        active_targets: set[tuple[int, int]],
        blocked_cells: set[tuple[int, int]],
    ) -> tuple[int, int]:
        candidates = [
            cell
            for cell in self.grid.neighbors_with_wait(start)
            if cell not in blocked_cells
        ]
        if not candidates:
            return start

        center = (self.grid.width // 2, self.grid.height // 2)
        return min(
            candidates,
            key=lambda cell: (
                1 if cell in active_targets else 0,
                1 if cell == start else 0,
                GridMap.manhattan(cell, center),
                cell[1],
                cell[0],
            ),
        )

    def _can_reserve_path(
        self,
        path: list[PlanStep],
        reservation_table: ReservationTable,
        robot_id: str,
    ) -> bool:
        for step in path:
            if reservation_table.is_vertex_reserved(step.time, step.position, robot_id):
                return False
        for current, nxt in zip(path, path[1:]):
            if reservation_table.is_edge_reserved(
                current.time, current.position, nxt.position, robot_id
            ):
                return False
            if reservation_table.is_edge_reserved(
                current.time, nxt.position, current.position, robot_id
            ):
                return False
        return True

    def _select_safe_next_step(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        preferred_next: tuple[int, int],
        reservation_table: ReservationTable,
        tick: int,
        blocked_cells: set[tuple[int, int]],
    ) -> tuple[int, int]:
        candidates = [preferred_next]
        others = sorted(
            self.grid.neighbors_with_wait(start),
            key=lambda pos: (GridMap.manhattan(pos, target), pos[1], pos[0]),
        )
        for cand in others:
            if cand in blocked_cells:
                continue
            if cand not in candidates:
                candidates.append(cand)

        for cand in candidates:
            if cand in blocked_cells:
                continue
            if reservation_table.is_vertex_reserved(tick + 1, cand):
                continue
            if reservation_table.is_edge_reserved(tick, cand, start):
                continue
            return cand

        return start

    def _priority_key(
        self,
        robot: RobotState,
        tasks: dict[str, TaskState],
        blocking_score: int,
        wait_streak: int,
        tick: int,
        total_robots: int,
        rank: int,
    ) -> tuple[int, int, int, int, int]:
        target = self._target_for_robot(robot, tasks)
        active_task_flag = 0 if (target is not None or blocking_score > 0) else 1
        rotation_rank = (rank - (tick % max(total_robots, 1))) % max(total_robots, 1)
        eta = GridMap.manhattan(robot.position, target) if target is not None else 10**9
        return (active_task_flag, -blocking_score, -wait_streak, rotation_rank, eta)

    @staticmethod
    def _target_for_robot(
        robot: RobotState,
        tasks: dict[str, TaskState],
    ) -> tuple[int, int] | None:
        if robot.current_task_id is None:
            return None
        task = tasks[robot.current_task_id]
        if robot.phase == "to_pickup":
            return task.pickup
        if robot.phase == "to_dropoff":
            return task.dropoff
        return None
