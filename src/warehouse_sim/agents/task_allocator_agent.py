from __future__ import annotations

from warehouse_sim.grid import GridMap
from warehouse_sim.models import AllocationPolicy, Assignment, RobotState, TaskState
from warehouse_sim.optimization.hungarian import hungarian_assignment


class TaskAllocatorAgent:
    """Task allocator with greedy and Hungarian policies."""

    def assign_tasks(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        grid: GridMap,
        tick: int,
        policy: AllocationPolicy = "greedy",
    ) -> list[Assignment]:
        del grid

        available_tasks = sorted(
            [task for task in tasks.values() if task.is_available(tick)],
            key=lambda task: (task.release_tick, task.task_id),
        )
        idle_robot_ids = sorted(
            [robot.robot_id for robot in robots.values() if robot.current_task_id is None]
        )

        if not available_tasks or not idle_robot_ids:
            return []

        if policy == "hungarian":
            return self._assign_hungarian(robots, available_tasks, idle_robot_ids, tick)
        return self._assign_greedy(robots, available_tasks, idle_robot_ids, tick)

    def _assign_greedy(
        self,
        robots: dict[str, RobotState],
        available_tasks: list[TaskState],
        idle_robot_ids: list[str],
        tick: int,
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        free_robots = set(idle_robot_ids)

        for task in available_tasks:
            if not free_robots:
                break

            best_robot_id = min(
                free_robots,
                key=lambda rid: (
                    GridMap.manhattan(robots[rid].position, task.pickup),
                    rid,
                ),
            )
            eta = GridMap.manhattan(robots[best_robot_id].position, task.pickup)
            self._apply_assignment(
                robots=robots,
                task=task,
                robot_id=best_robot_id,
                tick=tick,
            )
            assignments.append(
                Assignment(
                    robot_id=best_robot_id,
                    task_id=task.task_id,
                    eta_to_pickup=eta,
                )
            )
            free_robots.remove(best_robot_id)

        return assignments

    def _assign_hungarian(
        self,
        robots: dict[str, RobotState],
        available_tasks: list[TaskState],
        idle_robot_ids: list[str],
        tick: int,
    ) -> list[Assignment]:
        # Deterministic order is guaranteed by sorted robot/task IDs.
        tasks = sorted(available_tasks, key=lambda t: t.task_id)
        cost_matrix: list[list[float]] = []
        for robot_id in idle_robot_ids:
            row: list[float] = []
            for task in tasks:
                row.append(float(GridMap.manhattan(robots[robot_id].position, task.pickup)))
            cost_matrix.append(row)

        assignment_idx, _ = hungarian_assignment(cost_matrix)

        assignments: list[Assignment] = []
        for row_idx, col_idx in enumerate(assignment_idx):
            if col_idx < 0 or col_idx >= len(tasks):
                continue

            robot_id = idle_robot_ids[row_idx]
            task = tasks[col_idx]
            if task.assigned_robot_id is not None:
                continue

            eta = int(cost_matrix[row_idx][col_idx])
            self._apply_assignment(
                robots=robots,
                task=task,
                robot_id=robot_id,
                tick=tick,
            )
            assignments.append(
                Assignment(
                    robot_id=robot_id,
                    task_id=task.task_id,
                    eta_to_pickup=eta,
                )
            )

        return assignments

    @staticmethod
    def _apply_assignment(
        robots: dict[str, RobotState],
        task: TaskState,
        robot_id: str,
        tick: int,
    ) -> None:
        robots[robot_id].current_task_id = task.task_id
        robots[robot_id].phase = "to_pickup"
        task.assigned_robot_id = robot_id
        task.assigned_tick = tick
