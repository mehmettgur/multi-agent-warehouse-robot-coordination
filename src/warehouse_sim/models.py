from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Position = tuple[int, int]
Mode = Literal["baseline", "coordinated"]
Phase = Literal["to_pickup", "to_dropoff"]
ActionType = Literal["MOVE", "WAIT"]


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    start: Position


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    pickup: Position
    dropoff: Position
    release_tick: int = 0


@dataclass
class TaskState:
    task_id: str
    pickup: Position
    dropoff: Position
    release_tick: int = 0
    assigned_robot_id: str | None = None
    assigned_tick: int | None = None
    pickup_tick: int | None = None
    completed_tick: int | None = None

    @classmethod
    def from_spec(cls, spec: TaskSpec) -> "TaskState":
        return cls(
            task_id=spec.task_id,
            pickup=spec.pickup,
            dropoff=spec.dropoff,
            release_tick=spec.release_tick,
        )

    def is_available(self, tick: int) -> bool:
        return (
            self.release_tick <= tick
            and self.assigned_robot_id is None
            and self.completed_tick is None
        )

    def is_completed(self) -> bool:
        return self.completed_tick is not None


@dataclass
class RobotState:
    robot_id: str
    position: Position
    current_task_id: str | None = None
    phase: Phase | None = None
    last_action: ActionType = "WAIT"


@dataclass(frozen=True)
class PlanStep:
    position: Position
    time: int


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    target: Position


@dataclass(frozen=True)
class Assignment:
    robot_id: str
    task_id: str
    eta_to_pickup: int


@dataclass
class SimulationConfig:
    name: str
    seed: int
    width: int
    height: int
    obstacles: set[Position]
    stations: dict[str, list[Position]]
    robots: list[RobotSpec]
    tasks: list[TaskSpec]
    max_ticks: int = 200
    events: list[dict] = field(default_factory=list)


@dataclass
class TickSnapshot:
    tick: int
    robot_positions: dict[str, Position]
    robot_tasks: dict[str, str | None]
    collision_events: int
    completed_tasks: int

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "robot_positions": {
                rid: [pos[0], pos[1]] for rid, pos in self.robot_positions.items()
            },
            "robot_tasks": self.robot_tasks,
            "collision_events": self.collision_events,
            "completed_tasks": self.completed_tasks,
        }


@dataclass
class MetricsReport:
    makespan: int
    total_path_length: int
    avg_task_completion_time: float
    wait_count: int
    collision_count: int
    replanning_count: int
    completed_tasks: int
    total_tasks: int
    task_completion_times: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "makespan": self.makespan,
            "total_path_length": self.total_path_length,
            "avg_task_completion_time": self.avg_task_completion_time,
            "wait_count": self.wait_count,
            "collision_count": self.collision_count,
            "replanning_count": self.replanning_count,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "task_completion_times": self.task_completion_times,
        }


@dataclass
class RunResult:
    scenario_name: str
    mode: Mode
    seed: int
    metrics: MetricsReport
    timeline: list[TickSnapshot]

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "mode": self.mode,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "timeline": [snapshot.to_dict() for snapshot in self.timeline],
        }
