from __future__ import annotations

from warehouse_sim.models import MetricsReport, TaskState


class MetricsCollector:
    def __init__(self) -> None:
        self.total_path_length = 0
        self.wait_count = 0
        self.collision_count = 0
        self.replanning_count = 0

    def add_move(self) -> None:
        self.total_path_length += 1

    def add_wait(self) -> None:
        self.wait_count += 1

    def add_collisions(self, collisions: int) -> None:
        self.collision_count += collisions

    def add_replans(self, replans: int) -> None:
        self.replanning_count += replans

    def finalize(
        self,
        tasks: dict[str, TaskState],
        elapsed_ticks: int,
    ) -> MetricsReport:
        completion_times: dict[str, int] = {}
        for task in tasks.values():
            if task.completed_tick is None:
                continue
            completion_times[task.task_id] = task.completed_tick - task.release_tick

        avg_time = (
            sum(completion_times.values()) / len(completion_times)
            if completion_times
            else 0.0
        )

        return MetricsReport(
            makespan=elapsed_ticks,
            total_path_length=self.total_path_length,
            avg_task_completion_time=round(avg_time, 3),
            wait_count=self.wait_count,
            collision_count=self.collision_count,
            replanning_count=self.replanning_count,
            completed_tasks=len(completion_times),
            total_tasks=len(tasks),
            task_completion_times=completion_times,
        )
