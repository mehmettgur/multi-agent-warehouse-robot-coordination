from __future__ import annotations

import random
from itertools import combinations

from warehouse_sim.agents.task_allocator_agent import TaskAllocatorAgent
from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.events import EventEngine
from warehouse_sim.grid import GridMap
from warehouse_sim.metrics import MetricsCollector
from warehouse_sim.models import (
    AllocationPolicy,
    Mode,
    PlanStep,
    PlannerConfig,
    PlannerDiagnostics,
    RobotState,
    RunResult,
    TaskState,
    TickSnapshot,
)
from warehouse_sim.pathfinding import plan_path_space_time
from warehouse_sim.policies.baseline_policy import BaselinePolicy
from warehouse_sim.policies.coordinated_policy import CoordinatedPolicy
from warehouse_sim.reservation import ReservationTable


class CoordinatorAgent:
    def __init__(
        self,
        scenario_name: str,
        mode: Mode,
        seed: int,
        grid: GridMap,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        max_ticks: int,
        planner: PlannerConfig,
        allocator_policy: AllocationPolicy,
        events: list[dict],
    ) -> None:
        self.scenario_name = scenario_name
        self.mode = mode
        self.seed = seed
        self.grid = grid
        self.robots = robots
        self.tasks = tasks
        self.max_ticks = max_ticks
        self.planner = planner
        self.allocator_policy = allocator_policy

        self.rng = random.Random(seed)
        self.metrics = MetricsCollector()
        self.allocator = TaskAllocatorAgent()
        self.traffic_manager = TrafficManagerAgent(
            grid=grid,
            max_ticks=max_ticks,
            planner=planner,
        )
        self.baseline_policy = BaselinePolicy(grid=grid, max_ticks=max_ticks, planner=planner)
        self.coordinated_policy = CoordinatedPolicy(traffic_manager=self.traffic_manager)
        self.wait_streaks = {robot_id: 0 for robot_id in robots}
        self.tasks_completed_per_robot = {robot_id: 0 for robot_id in robots}
        self.event_engine = EventEngine(events)
        self._last_blocked_cells: set[tuple[int, int]] = set()

    def run(self) -> RunResult:
        timeline: list[TickSnapshot] = []
        elapsed_ticks = 0

        for tick in range(self.max_ticks):
            elapsed_ticks = tick
            active_blocked = self.event_engine.active_blocked_cells(tick)
            if (
                self.mode == "coordinated"
                and active_blocked != self._last_blocked_cells
                and active_blocked
            ):
                self.metrics.add_dynamic_block_replans(len(self.robots))
            self._last_blocked_cells = set(active_blocked)

            self._update_task_progress_on_position(tick)

            alloc_policy = "greedy" if self.mode == "baseline" else self.allocator_policy
            self.allocator.assign_tasks(
                robots=self.robots,
                tasks=self.tasks,
                grid=self.grid,
                tick=tick,
                policy=alloc_policy,
            )
            self._opportunistic_reassignment(tick)
            self._update_task_progress_on_position(tick)

            if self.mode == "coordinated":
                intents, _, diagnostics = self.coordinated_policy.plan_intents(
                    robots=self.robots,
                    tasks=self.tasks,
                    tick=tick,
                    wait_streaks=self.wait_streaks,
                    blocked_cells=active_blocked,
                    planner=self.planner,
                )
                self.metrics.add_general_replans(len(self.robots))
            else:
                intents, diagnostics = self.baseline_policy.plan_intents(
                    robots=self.robots,
                    tasks=self.tasks,
                    tick=tick,
                    blocked_cells=active_blocked,
                    planner=self.planner,
                )

            self.metrics.add_planner_diagnostics(diagnostics)

            delayed_robots, delay_event_count = self.event_engine.delayed_robots(
                tick=tick,
                robot_ids=sorted(self.robots),
                rng=self.rng,
            )
            self.metrics.add_delay_events(delay_event_count)
            for robot_id in delayed_robots:
                intents[robot_id] = self.robots[robot_id].position

            for robot_id, target in list(intents.items()):
                if target in active_blocked:
                    intents[robot_id] = self.robots[robot_id].position

            conflicts, conflict_pairs = self._detect_conflicts(intents)
            if self.mode == "coordinated" and conflicts:
                intents, micro_replans, micro_diagnostics = self._resolve_local_conflicts(
                    intents=intents,
                    conflicts=conflicts,
                    tick=tick,
                    active_blocked=active_blocked,
                )
                self.metrics.add_micro_replans(micro_replans)
                self.metrics.add_planner_diagnostics(micro_diagnostics)

                conflicts, conflict_pairs = self._detect_conflicts(intents)
                if conflicts:
                    intents = self._deterministic_fallback(intents=intents, conflicts=conflicts, tick=tick)
                    conflicts, conflict_pairs = self._detect_conflicts(intents)

            if conflict_pairs:
                if self.mode == "baseline":
                    self.metrics.add_collisions(conflict_pairs)
                else:
                    # Coordinated mode converts unresolved conflicts into WAIT actions.
                    conflict_pairs = 0

            self._apply_intents(intents=intents, conflicts=conflicts)
            self._update_task_progress_on_position(tick + 1)

            timeline.append(
                TickSnapshot(
                    tick=tick + 1,
                    robot_positions={
                        rid: robot.position
                        for rid, robot in sorted(self.robots.items())
                    },
                    robot_tasks={
                        rid: robot.current_task_id
                        for rid, robot in sorted(self.robots.items())
                    },
                    collision_events=conflict_pairs,
                    completed_tasks=sum(
                        1 for task in self.tasks.values() if task.is_completed()
                    ),
                    blocked_cells=sorted(active_blocked),
                )
            )

            if all(task.is_completed() for task in self.tasks.values()):
                elapsed_ticks = tick + 1
                break

        report = self.metrics.finalize(
            tasks=self.tasks,
            elapsed_ticks=elapsed_ticks,
            tasks_completed_per_robot=self.tasks_completed_per_robot,
        )
        return RunResult(
            scenario_name=self.scenario_name,
            mode=self.mode,
            seed=self.seed,
            metrics=report,
            timeline=timeline,
            planner_algorithm=self.planner.algorithm,
            heuristic_weight=self.planner.heuristic_weight,
            allocator_policy=self.allocator_policy,
        )

    def _update_task_progress_on_position(self, tick: int) -> None:
        for robot in self.robots.values():
            if robot.current_task_id is None:
                continue
            task = self.tasks[robot.current_task_id]
            if robot.phase == "to_pickup" and robot.position == task.pickup:
                robot.phase = "to_dropoff"
                if task.pickup_tick is None:
                    task.pickup_tick = tick
                continue
            if robot.phase == "to_dropoff" and robot.position == task.dropoff:
                task.completed_tick = tick
                task.completed_by_robot_id = robot.robot_id
                task.assigned_robot_id = None
                self.tasks_completed_per_robot[robot.robot_id] = (
                    self.tasks_completed_per_robot.get(robot.robot_id, 0) + 1
                )
                robot.current_task_id = None
                robot.phase = None

    def _opportunistic_reassignment(self, tick: int) -> None:
        for task in sorted(self.tasks.values(), key=lambda item: item.task_id):
            if task.completed_tick is not None:
                continue
            if task.assigned_robot_id is None:
                continue
            if task.pickup_tick is not None:
                continue

            assigned_robot = self.robots[task.assigned_robot_id]
            if assigned_robot.position == task.pickup:
                continue

            candidates = sorted(
                [
                    robot.robot_id
                    for robot in self.robots.values()
                    if robot.current_task_id is None and robot.position == task.pickup
                ]
            )
            if not candidates:
                continue

            new_robot_id = candidates[0]
            old_robot_id = task.assigned_robot_id

            if (
                old_robot_id is not None
                and self.robots[old_robot_id].current_task_id == task.task_id
            ):
                self.robots[old_robot_id].current_task_id = None
                self.robots[old_robot_id].phase = None

            task.assigned_robot_id = new_robot_id
            task.assigned_tick = tick
            self.robots[new_robot_id].current_task_id = task.task_id
            self.robots[new_robot_id].phase = "to_pickup"

    def _detect_conflicts(
        self,
        intents: dict[str, tuple[int, int]],
    ) -> tuple[set[str], int]:
        conflicts: set[str] = set()
        collision_pairs = 0

        target_to_robots: dict[tuple[int, int], list[str]] = {}
        for robot_id, target in intents.items():
            target_to_robots.setdefault(target, []).append(robot_id)

        for robot_ids in target_to_robots.values():
            if len(robot_ids) <= 1:
                continue
            collision_pairs += len(robot_ids) * (len(robot_ids) - 1) // 2
            conflicts.update(robot_ids)

        for rid_a, rid_b in combinations(sorted(intents), 2):
            a_current = self.robots[rid_a].position
            b_current = self.robots[rid_b].position
            a_next = intents[rid_a]
            b_next = intents[rid_b]
            if a_next == b_current and b_next == a_current and a_current != b_current:
                collision_pairs += 1
                conflicts.add(rid_a)
                conflicts.add(rid_b)

        return conflicts, collision_pairs

    def _resolve_local_conflicts(
        self,
        intents: dict[str, tuple[int, int]],
        conflicts: set[str],
        tick: int,
        active_blocked: set[tuple[int, int]],
    ) -> tuple[dict[str, tuple[int, int]], int, list[PlannerDiagnostics]]:
        resolved = dict(intents)
        micro_replans = 0
        diagnostics: list[PlannerDiagnostics] = []

        if not conflicts:
            return resolved, micro_replans, diagnostics

        reservation = ReservationTable()
        for robot_id in sorted(self.robots):
            if robot_id in conflicts:
                continue
            current = self.robots[robot_id].position
            preferred_next = resolved[robot_id]
            next_pos, _ = self._reserve_best_one_step(
                reservation=reservation,
                robot_id=robot_id,
                current=current,
                preferred_next=preferred_next,
                target=preferred_next,
                tick=tick,
                blocked_cells=active_blocked,
            )
            resolved[robot_id] = next_pos

        for robot_id in sorted(conflicts, key=self._fallback_priority):
            robot = self.robots[robot_id]
            target = self.traffic_manager._target_for_robot(robot, self.tasks)
            if target is None:
                next_pos, _ = self._reserve_best_one_step(
                    reservation=reservation,
                    robot_id=robot_id,
                    current=robot.position,
                    preferred_next=robot.position,
                    target=robot.position,
                    tick=tick,
                    blocked_cells=active_blocked,
                )
                resolved[robot_id] = next_pos
                continue

            max_time = min(self.max_ticks, tick + 6)
            path, diag = plan_path_space_time(
                grid=self.grid,
                start=robot.position,
                goal=target,
                start_time=tick,
                max_time=max_time,
                planner=self.planner,
                reservations=reservation,
                robot_id=robot_id,
                blocked_cells=active_blocked,
            )
            diagnostics.append(diag)
            micro_replans += 1

            if path is None or len(path) < 2:
                preferred_next = robot.position
            else:
                preferred_next = path[1].position

            next_pos, _ = self._reserve_best_one_step(
                reservation=reservation,
                robot_id=robot_id,
                current=robot.position,
                preferred_next=preferred_next,
                target=target,
                tick=tick,
                blocked_cells=active_blocked,
            )
            resolved[robot_id] = next_pos

        return resolved, micro_replans, diagnostics

    def _reserve_best_one_step(
        self,
        reservation: ReservationTable,
        robot_id: str,
        current: tuple[int, int],
        preferred_next: tuple[int, int],
        target: tuple[int, int],
        tick: int,
        blocked_cells: set[tuple[int, int]],
    ) -> tuple[tuple[int, int], bool]:
        candidates: list[tuple[int, int]] = []

        for candidate in [preferred_next, current]:
            if candidate not in candidates:
                candidates.append(candidate)

        for neighbor in sorted(
            self.grid.neighbors_with_wait(current),
            key=lambda pos: (GridMap.manhattan(pos, target), pos[1], pos[0]),
        ):
            if neighbor not in candidates:
                candidates.append(neighbor)

        for candidate in candidates:
            if candidate in blocked_cells:
                continue
            one_step = [
                PlanStep(position=current, time=tick),
                PlanStep(position=candidate, time=tick + 1),
            ]
            try:
                reservation.reserve_path(robot_id, one_step)
                return candidate, True
            except ValueError:
                continue

        return current, False

    def _deterministic_fallback(
        self,
        intents: dict[str, tuple[int, int]],
        conflicts: set[str],
        tick: int,
    ) -> dict[str, tuple[int, int]]:
        resolved = dict(intents)
        if not conflicts:
            return resolved

        max_rounds = max(4, len(self.robots) * 4)
        for _ in range(max_rounds):
            active_conflicts, _ = self._detect_conflicts(resolved)
            if not active_conflicts:
                del tick
                return resolved

            loser = max(active_conflicts, key=self._fallback_priority)
            resolved[loser] = self.robots[loser].position

        # Hard safety: if anything still conflicts, force all conflicting robots to wait.
        final_conflicts, _ = self._detect_conflicts(resolved)
        for robot_id in sorted(final_conflicts):
            resolved[robot_id] = self.robots[robot_id].position

        del tick
        return resolved

    def _fallback_priority(self, robot_id: str) -> tuple[int, int, int, str]:
        robot = self.robots[robot_id]
        has_task = 0 if robot.current_task_id is not None else 1
        wait_streak = -self.wait_streaks.get(robot_id, 0)
        target = self.traffic_manager._target_for_robot(robot, self.tasks)
        eta = GridMap.manhattan(robot.position, target) if target is not None else 10**9
        return (has_task, wait_streak, eta, robot_id)

    def _apply_intents(self, intents: dict[str, tuple[int, int]], conflicts: set[str]) -> None:
        for robot_id in sorted(self.robots):
            robot = self.robots[robot_id]
            intended = intents[robot_id]

            if robot_id in conflicts or intended == robot.position:
                robot.last_action = "WAIT"
                self.wait_streaks[robot_id] = self.wait_streaks.get(robot_id, 0) + 1
                self.metrics.add_wait()
                self.metrics.add_cell_visit(robot.position)
                continue

            robot.position = intended
            robot.last_action = "MOVE"
            self.wait_streaks[robot_id] = 0
            self.metrics.add_move()
            self.metrics.add_cell_visit(robot.position)
