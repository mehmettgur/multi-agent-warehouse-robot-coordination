from __future__ import annotations

from warehouse_sim.agents.coordinator_agent import CoordinatorAgent
from warehouse_sim.grid import GridMap
from warehouse_sim.models import (
    AllocationPolicy,
    Mode,
    PlannerConfig,
    RobotState,
    RunResult,
    SimulationConfig,
    TaskState,
)


def run_simulation(
    config: SimulationConfig,
    mode: Mode,
    seed: int | None = None,
    planner_override: PlannerConfig | None = None,
    allocator_override: AllocationPolicy | None = None,
) -> RunResult:
    run_seed = config.seed if seed is None else seed
    planner = planner_override or config.planner

    if allocator_override is not None:
        allocator_policy = allocator_override
    elif mode == "baseline":
        allocator_policy = "greedy"
    else:
        allocator_policy = config.allocator_policy

    grid = GridMap(width=config.width, height=config.height, obstacles=set(config.obstacles))
    robots = {
        spec.robot_id: RobotState(robot_id=spec.robot_id, position=spec.start)
        for spec in sorted(config.robots, key=lambda r: r.robot_id)
    }
    tasks = {
        spec.task_id: TaskState.from_spec(spec)
        for spec in sorted(config.tasks, key=lambda t: t.task_id)
    }

    coordinator = CoordinatorAgent(
        scenario_name=config.name,
        mode=mode,
        seed=run_seed,
        grid=grid,
        robots=robots,
        tasks=tasks,
        max_ticks=config.max_ticks,
        planner=planner,
        allocator_policy=allocator_policy,
        events=config.events,
    )
    return coordinator.run()
