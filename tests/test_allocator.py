from warehouse_sim.agents.task_allocator_agent import TaskAllocatorAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.models import RobotState, TaskState


def test_greedy_allocator_assigns_nearest_idle_robots() -> None:
    grid = GridMap(width=8, height=4, obstacles=set())
    robots = {
        "R1": RobotState(robot_id="R1", position=(0, 0)),
        "R2": RobotState(robot_id="R2", position=(5, 0)),
    }
    tasks = {
        "T1": TaskState(task_id="T1", pickup=(1, 0), dropoff=(7, 0), release_tick=0),
        "T2": TaskState(task_id="T2", pickup=(4, 0), dropoff=(0, 0), release_tick=0),
    }

    allocator = TaskAllocatorAgent()
    assignments = allocator.assign_tasks(robots=robots, tasks=tasks, grid=grid, tick=0)

    by_task = {assignment.task_id: assignment.robot_id for assignment in assignments}
    assert by_task["T1"] == "R1"
    assert by_task["T2"] == "R2"
