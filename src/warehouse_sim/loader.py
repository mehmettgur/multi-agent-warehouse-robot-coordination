from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.models import PlannerConfig, RobotSpec, SimulationConfig, TaskSpec


def load_scenario(path: str | Path) -> SimulationConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    obstacles = {tuple(cell) for cell in data["grid"]["obstacles"]}
    stations = {
        key: [tuple(cell) for cell in value]
        for key, value in data.get("stations", {}).items()
    }
    robots = [
        RobotSpec(robot_id=robot["id"], start=tuple(robot["start"]))
        for robot in data["robots"]
    ]
    tasks = [
        TaskSpec(
            task_id=task["id"],
            pickup=tuple(task["pickup"]),
            dropoff=tuple(task["dropoff"]),
            release_tick=task.get("release_tick", 0),
        )
        for task in data["tasks"]
    ]

    planner_raw = data.get("planner", {})
    planner = PlannerConfig(
        algorithm=planner_raw.get("algorithm", "astar"),
        heuristic_weight=float(planner_raw.get("heuristic_weight", 1.4)),
    )

    return SimulationConfig(
        name=data["name"],
        seed=data.get("seed", 0),
        width=data["grid"]["width"],
        height=data["grid"]["height"],
        obstacles=obstacles,
        stations=stations,
        robots=robots,
        tasks=tasks,
        max_ticks=data.get("simulation", {}).get("max_ticks", 200),
        events=data.get("events", []),
        planner=planner,
        allocator_policy=data.get("allocator_policy", "hungarian"),
    )
