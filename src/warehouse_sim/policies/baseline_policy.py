from __future__ import annotations

from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.models import Position, RobotState, TaskState
from warehouse_sim.pathfinding import astar_space_time


class BaselinePolicy:
    """Independent greedy planning without centralized reservations."""

    def __init__(self, grid: GridMap, max_ticks: int) -> None:
        self.grid = grid
        self.max_ticks = max_ticks

    def plan_intents(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        tick: int,
    ) -> dict[str, Position]:
        intents: dict[str, Position] = {}

        for robot_id in sorted(robots):
            robot = robots[robot_id]
            target = TrafficManagerAgent._target_for_robot(robot, tasks)
            if target is None:
                intents[robot_id] = robot.position
                continue

            max_time = min(self.max_ticks, tick + self.grid.width * self.grid.height * 2)
            path = astar_space_time(
                grid=self.grid,
                start=robot.position,
                goal=target,
                start_time=tick,
                max_time=max_time,
                reservations=None,
                robot_id=robot_id,
            )
            if path is None or len(path) < 2:
                intents[robot_id] = robot.position
            else:
                intents[robot_id] = path[1].position

        return intents
