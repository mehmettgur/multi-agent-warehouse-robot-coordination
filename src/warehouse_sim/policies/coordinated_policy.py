from __future__ import annotations

from warehouse_sim.agents.robot_agent import RobotAgent
from warehouse_sim.agents.traffic_manager_agent import TrafficManagerAgent
from warehouse_sim.models import Position, RobotState, TaskState
from warehouse_sim.reservation import ReservationTable


class CoordinatedPolicy:
    def __init__(self, traffic_manager: TrafficManagerAgent) -> None:
        self.traffic_manager = traffic_manager
        self.robot_agents: dict[str, RobotAgent] = {}

    def plan_intents(
        self,
        robots: dict[str, RobotState],
        tasks: dict[str, TaskState],
        tick: int,
        wait_streaks: dict[str, int] | None = None,
    ) -> tuple[dict[str, Position], ReservationTable]:
        planned_paths, reservation_table = self.traffic_manager.plan_and_reserve(
            robots=robots,
            tasks=tasks,
            tick=tick,
            wait_streaks=wait_streaks,
        )

        intents: dict[str, Position] = {}
        for robot_id, path in planned_paths.items():
            agent = self.robot_agents.setdefault(robot_id, RobotAgent(robot_id))
            action = agent.next_action(robots[robot_id], path, tick)
            intents[robot_id] = action.target

        return intents, reservation_table
