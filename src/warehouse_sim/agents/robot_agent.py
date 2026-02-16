from __future__ import annotations

from warehouse_sim.models import Action, PlanStep, RobotState


class RobotAgent:
    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id

    def next_action(
        self,
        robot: RobotState,
        planned_path: list[PlanStep],
        tick: int,
    ) -> Action:
        if len(planned_path) >= 2 and planned_path[1].time == tick + 1:
            target = planned_path[1].position
            if target != robot.position:
                return Action(action_type="MOVE", target=target)
        return Action(action_type="WAIT", target=robot.position)
