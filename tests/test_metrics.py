from warehouse_sim.metrics import MetricsCollector
from warehouse_sim.models import TaskState


def test_metrics_finalize_computes_expected_aggregates() -> None:
    metrics = MetricsCollector()
    for _ in range(5):
        metrics.add_move()
    for _ in range(2):
        metrics.add_wait()
    metrics.add_collisions(1)
    metrics.add_replans(3)

    tasks = {
        "T1": TaskState(
            task_id="T1",
            pickup=(0, 0),
            dropoff=(1, 1),
            release_tick=0,
            completed_tick=6,
        ),
        "T2": TaskState(
            task_id="T2",
            pickup=(2, 2),
            dropoff=(3, 3),
            release_tick=2,
            completed_tick=None,
        ),
    }

    report = metrics.finalize(tasks=tasks, elapsed_ticks=7)

    assert report.makespan == 7
    assert report.total_path_length == 5
    assert report.wait_count == 2
    assert report.collision_count == 1
    assert report.replanning_count == 3
    assert report.completed_tasks == 1
    assert report.total_tasks == 2
    assert report.avg_task_completion_time == 6.0
    assert report.task_completion_times == {"T1": 6}
