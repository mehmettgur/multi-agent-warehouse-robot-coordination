from warehouse_sim.metrics import MetricsCollector
from warehouse_sim.models import PlannerDiagnostics, TaskState


def test_metrics_finalize_computes_extended_aggregates() -> None:
    metrics = MetricsCollector()
    for _ in range(5):
        metrics.add_move()
    for _ in range(2):
        metrics.add_wait()
    metrics.add_collisions(1)
    metrics.add_general_replans(3)
    metrics.add_micro_replans(2)
    metrics.add_delay_events(4)
    metrics.add_dynamic_block_replans(1)
    metrics.add_cell_visit((0, 0))
    metrics.add_cell_visit((0, 0))
    metrics.add_cell_visit((1, 0))
    metrics.add_planner_diagnostics(
        [
            PlannerDiagnostics("astar", 10, 1.2, 5, True),
            PlannerDiagnostics("dijkstra", 12, 1.8, 6, True),
        ]
    )

    tasks = {
        "T1": TaskState(
            task_id="T1",
            pickup=(0, 0),
            dropoff=(1, 1),
            release_tick=0,
            completed_tick=6,
            completed_by_robot_id="R1",
        ),
        "T2": TaskState(
            task_id="T2",
            pickup=(2, 2),
            dropoff=(3, 3),
            release_tick=2,
            completed_tick=None,
        ),
    }

    report = metrics.finalize(
        tasks=tasks,
        elapsed_ticks=7,
        tasks_completed_per_robot={"R1": 1, "R2": 0},
    )

    assert report.makespan == 7
    assert report.total_path_length == 5
    assert report.wait_count == 2
    assert report.collision_count == 1
    assert report.replanning_count == 5
    assert report.general_replanning_count == 3
    assert report.micro_replanning_count == 2
    assert report.delay_event_count == 4
    assert report.dynamic_block_replans == 1
    assert report.completed_tasks == 1
    assert report.total_tasks == 2
    assert report.avg_task_completion_time == 6.0
    assert report.task_completion_times == {"T1": 6}
    assert report.throughput > 0
    assert report.tasks_completed_per_robot == {"R1": 1, "R2": 0}
    assert report.congestion_heatmap["0,0"] == 2
    assert report.planner_expanded_nodes_total == 22
