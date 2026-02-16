from __future__ import annotations

import random
from itertools import combinations

from warehouse_sim.agents.task_allocator_agent import TaskAllocatorAgent
from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.metrics import MetricsCollector
from warehouse_sim.models import Mode, RobotState, RunResult, TaskState, TickSnapshot
from warehouse_sim.policies.baseline_policy import BaselinePolicy
from warehouse_sim.policies.coordinated_policy import CoordinatedPolicy


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
    ) -> None:
        self.scenario_name = scenario_name
        self.mode = mode
        self.seed = seed
        self.grid = grid
        self.robots = robots
        self.tasks = tasks
        self.max_ticks = max_ticks

        self.rng = random.Random(seed)
        self.metrics = MetricsCollector()
        self.allocator = TaskAllocatorAgent()
        self.traffic_manager = TrafficManagerAgent(grid=grid, max_ticks=max_ticks)
        self.baseline_policy = BaselinePolicy(grid=grid, max_ticks=max_ticks)
        self.coordinated_policy = CoordinatedPolicy(traffic_manager=self.traffic_manager)

    def run(self) -> RunResult:
        timeline: list[TickSnapshot] = []
        elapsed_ticks = 0

        for tick in range(self.max_ticks):
            elapsed_ticks = tick
            self._update_task_progress_on_position(tick)
            self.allocator.assign_tasks(
                robots=self.robots,
                tasks=self.tasks,
                grid=self.grid,
                tick=tick,
            )
            self._opportunistic_reassignment(tick)
            self._update_task_progress_on_position(tick)

            if self.mode == "coordinated":
                intents, _ = self.coordinated_policy.plan_intents(
                    robots=self.robots,
                    tasks=self.tasks,
                    tick=tick,
                )
                self.metrics.add_replans(len(self.robots))
            else:
                intents = self.baseline_policy.plan_intents(
                    robots=self.robots,
                    tasks=self.tasks,
                    tick=tick,
                )

            conflicts, conflict_pairs = self._detect_conflicts(intents)
            if conflict_pairs and self.mode == "baseline":
                self.metrics.add_collisions(conflict_pairs)

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
                )
            )

            if all(task.is_completed() for task in self.tasks.values()):
                elapsed_ticks = tick + 1
                break

        report = self.metrics.finalize(tasks=self.tasks, elapsed_ticks=elapsed_ticks)
        return RunResult(
            scenario_name=self.scenario_name,
            mode=self.mode,
            seed=self.seed,
            metrics=report,
            timeline=timeline,
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
                task.assigned_robot_id = None
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

    def _apply_intents(self, intents: dict[str, tuple[int, int]], conflicts: set[str]) -> None:
        for robot_id in sorted(self.robots):
            robot = self.robots[robot_id]
            intended = intents[robot_id]

            if robot_id in conflicts or intended == robot.position:
                robot.last_action = "WAIT"
                self.metrics.add_wait()
                continue

            robot.position = intended
            robot.last_action = "MOVE"
            self.metrics.add_move()
