from __future__ import annotations

from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.models import PlannerConfig, PlannerDiagnostics, Position, RobotState, TaskState
from warehouse_sim.pathfinding import plan_path_space_time


class BaselinePolicy:
    """Independent planning without centralized reservations."""

    def __init__(self, grid: GridMap, max_ticks: int, planner: PlannerConfig) -> None:
        self.grid = grid
        self.max_ticks = max_ticks
        self.planner = planner

    def plan_intents(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        tick: int,
        blocked_cells: set[tuple[int, int]] | None = None,
        planner: PlannerConfig | None = None,
    ) -> tuple[dict[str, Position], list[PlannerDiagnostics]]:
        intents: dict[str, Position] = {}
        diagnostics: list[PlannerDiagnostics] = []
        blocked = blocked_cells or set()
        planner_cfg = planner or self.planner

        for robot_id in sorted(robots):
            robot = robots[robot_id]
            target = TrafficManagerAgent._target_for_robot(robot, tasks)
            if target is None:
                intents[robot_id] = robot.position
                continue

            max_time = min(self.max_ticks, tick + self.grid.width * self.grid.height * 2)
            path, diag = plan_path_space_time(
                grid=self.grid,
                start=robot.position,
                goal=target,
                start_time=tick,
                max_time=max_time,
                planner=planner_cfg,
                reservations=None,
                robot_id=robot_id,
                blocked_cells=blocked,
            )
            diagnostics.append(diag)
            if path is None or len(path) < 2:
                intents[robot_id] = robot.position
            else:
                intents[robot_id] = path[1].position

        return intents, diagnostics
