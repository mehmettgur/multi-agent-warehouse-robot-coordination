from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Position = tuple[int, int]
Mode = Literal["baseline", "coordinated"]
Phase = Literal["to_pickup", "to_dropoff"]
ActionType = Literal["MOVE", "WAIT"]
PlannerAlgorithm = Literal["astar", "dijkstra", "weighted_astar"]
AllocationPolicy = Literal["greedy", "hungarian"]


@dataclass(frozen=True)
class PlannerConfig:
    algorithm: PlannerAlgorithm = "astar"
    heuristic_weight: float = 1.4


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
    completed_by_robot_id: str | None = None

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


@dataclass(frozen=True)
class PlannerDiagnostics:
    algorithm: PlannerAlgorithm
    expanded_nodes: int
    planning_time_ms: float
    path_cost: int
    found_path: bool

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "expanded_nodes": self.expanded_nodes,
            "planning_time_ms": self.planning_time_ms,
            "path_cost": self.path_cost,
            "found_path": self.found_path,
        }


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
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    allocator_policy: AllocationPolicy = "greedy"


@dataclass
class TickSnapshot:
    tick: int
    robot_positions: dict[str, Position]
    robot_tasks: dict[str, str | None]
    collision_events: int
    completed_tasks: int
    blocked_cells: list[Position] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "robot_positions": {
                rid: [pos[0], pos[1]] for rid, pos in self.robot_positions.items()
            },
            "robot_tasks": self.robot_tasks,
            "collision_events": self.collision_events,
            "completed_tasks": self.completed_tasks,
            "blocked_cells": [[x, y] for x, y in self.blocked_cells],
        }


@dataclass
class MetricsReport:
    makespan: int
    total_path_length: int
    avg_task_completion_time: float
    wait_count: int
    collision_count: int
    replanning_count: int
    general_replanning_count: int
    micro_replanning_count: int
    completed_tasks: int
    total_tasks: int
    task_completion_times: dict[str, int]
    throughput: float
    fairness_task_std: float
    fairness_task_cv: float
    tasks_completed_per_robot: dict[str, int]
    congestion_heatmap: dict[str, int]
    planner_expanded_nodes_total: int
    planner_time_ms_total: float
    planner_path_cost_total: int
    delay_event_count: int
    dynamic_block_replans: int

    def to_dict(self) -> dict:
        return {
            "makespan": self.makespan,
            "total_path_length": self.total_path_length,
            "avg_task_completion_time": self.avg_task_completion_time,
            "wait_count": self.wait_count,
            "collision_count": self.collision_count,
            "replanning_count": self.replanning_count,
            "general_replanning_count": self.general_replanning_count,
            "micro_replanning_count": self.micro_replanning_count,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "task_completion_times": self.task_completion_times,
            "throughput": self.throughput,
            "fairness_task_std": self.fairness_task_std,
            "fairness_task_cv": self.fairness_task_cv,
            "tasks_completed_per_robot": self.tasks_completed_per_robot,
            "congestion_heatmap": self.congestion_heatmap,
            "planner_expanded_nodes_total": self.planner_expanded_nodes_total,
            "planner_time_ms_total": self.planner_time_ms_total,
            "planner_path_cost_total": self.planner_path_cost_total,
            "delay_event_count": self.delay_event_count,
            "dynamic_block_replans": self.dynamic_block_replans,
        }


@dataclass
class RunResult:
    scenario_name: str
    mode: Mode
    seed: int
    metrics: MetricsReport
    timeline: list[TickSnapshot]
    planner_algorithm: PlannerAlgorithm
    heuristic_weight: float
    allocator_policy: AllocationPolicy

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "mode": self.mode,
            "seed": self.seed,
            "planner_algorithm": self.planner_algorithm,
            "heuristic_weight": self.heuristic_weight,
            "allocator_policy": self.allocator_policy,
            "metrics": self.metrics.to_dict(),
            "timeline": [snapshot.to_dict() for snapshot in self.timeline],
        }
