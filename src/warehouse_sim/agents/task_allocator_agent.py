from __future__ import annotations

from warehouse_sim.grid import GridMap
from warehouse_sim.models import Assignment, RobotState, TaskState


class TaskAllocatorAgent:
    """Greedy ETA-based task allocator."""

    def assign_tasks(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        grid: GridMap,
        tick: int,
    ) -> list[Assignment]:
        del grid  # Grid is available for future heuristics/extensions.

        available_tasks = sorted(
            [task for task in tasks.values() if task.is_available(tick)],
            key=lambda task: (task.release_tick, task.task_id),
        )
        idle_robot_ids = sorted(
            [
                robot.robot_id
                for robot in robots.values()
                if robot.current_task_id is None
            ]
        )

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

            robots[best_robot_id].current_task_id = task.task_id
            robots[best_robot_id].phase = "to_pickup"
            task.assigned_robot_id = best_robot_id
            task.assigned_tick = tick
            free_robots.remove(best_robot_id)
            assignments.append(
                Assignment(
                    robot_id=best_robot_id,
                    task_id=task.task_id,
                    eta_to_pickup=eta,
                )
            )

        return assignments
