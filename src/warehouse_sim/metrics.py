from __future__ import annotations

import math

from warehouse_sim.models import MetricsReport, PlannerDiagnostics, Position, TaskState


class MetricsCollector:
    def __init__(self) -> None:
        self.total_path_length = 0
        self.wait_count = 0
        self.collision_count = 0
        self.general_replanning_count = 0
        self.micro_replanning_count = 0
        self.delay_event_count = 0
        self.dynamic_block_replans = 0

        self._congestion: dict[Position, int] = {}

        self.planner_expanded_nodes_total = 0
        self.planner_time_ms_total = 0.0
        self.planner_path_cost_total = 0

    def add_move(self) -> None:
        self.total_path_length += 1

    def add_wait(self) -> None:
        self.wait_count += 1

    def add_collisions(self, collisions: int) -> None:
        self.collision_count += collisions

    def add_replans(self, replans: int) -> None:
        # Backward compatible alias for general replans.
        self.general_replanning_count += replans

    def add_general_replans(self, replans: int) -> None:
        self.general_replanning_count += replans

    def add_micro_replans(self, replans: int) -> None:
        self.micro_replanning_count += replans

    def add_delay_events(self, count: int) -> None:
        self.delay_event_count += count

    def add_dynamic_block_replans(self, count: int) -> None:
        self.dynamic_block_replans += count

    def add_cell_visit(self, pos: Position) -> None:
        self._congestion[pos] = self._congestion.get(pos, 0) + 1

    def add_planner_diagnostics(self, diagnostics: list[PlannerDiagnostics]) -> None:
        for diag in diagnostics:
            self.planner_expanded_nodes_total += diag.expanded_nodes
            self.planner_time_ms_total += diag.planning_time_ms
            self.planner_path_cost_total += diag.path_cost

    def finalize(
        self,
        tasks: dict[str, TaskState],
        elapsed_ticks: int,
        tasks_completed_per_robot: dict[str, int],
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

        total_tasks = len(tasks)
        completed_tasks = len(completion_times)
        throughput = completed_tasks / max(1, elapsed_ticks)

        counts = list(tasks_completed_per_robot.values())
        if not counts:
            fairness_std = 0.0
            fairness_cv = 0.0
        else:
            mean = sum(counts) / len(counts)
            variance = sum((value - mean) ** 2 for value in counts) / len(counts)
            fairness_std = math.sqrt(variance)
            fairness_cv = (fairness_std / mean) if mean > 0 else 0.0

        congestion_heatmap = {
            f"{x},{y}": count
            for (x, y), count in sorted(self._congestion.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        }

        replanning_count = self.general_replanning_count + self.micro_replanning_count

        return MetricsReport(
            makespan=elapsed_ticks,
            total_path_length=self.total_path_length,
            avg_task_completion_time=round(avg_time, 3),
            wait_count=self.wait_count,
            collision_count=self.collision_count,
            replanning_count=replanning_count,
            general_replanning_count=self.general_replanning_count,
            micro_replanning_count=self.micro_replanning_count,
            completed_tasks=completed_tasks,
            total_tasks=total_tasks,
            task_completion_times=completion_times,
            throughput=round(throughput, 4),
            fairness_task_std=round(fairness_std, 4),
            fairness_task_cv=round(fairness_cv, 4),
            tasks_completed_per_robot=dict(sorted(tasks_completed_per_robot.items())),
            congestion_heatmap=congestion_heatmap,
            planner_expanded_nodes_total=self.planner_expanded_nodes_total,
            planner_time_ms_total=round(self.planner_time_ms_total, 3),
            planner_path_cost_total=self.planner_path_cost_total,
            delay_event_count=self.delay_event_count,
            dynamic_block_replans=self.dynamic_block_replans,
        )
