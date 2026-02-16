# Multi-Agent Warehouse Robot Coordination

Deterministic 2D grid warehouse simulation for multi-agent task allocation and collision-free path planning.

## Features
- Agent-based architecture:
  - `RobotAgent`
  - `TaskAllocatorAgent` (greedy ETA)
  - `TrafficManagerAgent` (reservation table + prioritized planning)
  - `CoordinatorAgent` (tick loop + metrics)
- A* pathfinding with Manhattan heuristic and `WAIT` action.
- Coordinated mode with `vertex-time` + `edge-time` collision constraints.
- Deadlock-break priority scheme in coordinated mode:
  - wait-streak aware ordering
  - tick-based priority rotation
  - blocker-aware tie handling
- Baseline mode (independent no-reservation planning) for comparison.
- Scenario-driven experiments (JSON config).
- Metrics and reproducible runs with fixed seed.
- Minimal Streamlit dashboard.

## Project Structure
- `src/warehouse_sim/`: simulator source code
- `configs/scenarios/`: scenario JSON files
- `tests/`: unit and integration tests
- `app/dashboard.py`: web demo UI
- `results/`: generated run outputs

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,ui]
```

## Run Single Mode
```bash
python3 -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor.json \
  --mode coordinated \
  --seed 7
```

## Run Baseline vs Coordinated Comparison
```bash
python3 -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor.json \
  --seed 7 \
  --compare
```

This generates:
- `results/narrow_corridor_comparison.json`
- `results/narrow_corridor_comparison.csv`

## Run Tests
```bash
pytest
```

## Dashboard
```bash
streamlit run app/dashboard.py
```

Dashboard includes:
- Scenario selection
- Seed and mode controls
- Baseline vs coordinated comparison table
- Tick-based grid replay

## Core Metrics
- `makespan`
- `total_path_length`
- `avg_task_completion_time`
- `wait_count`
- `collision_count`
- `replanning_count`

## Acceptance Targets
- Coordinated mode: `collision_count == 0` in constrained scenarios.
- Baseline and coordinated run on same scenario and output comparable metrics.
- Same seed produces identical outputs.

## Implemented Scenarios
- `configs/scenarios/narrow_corridor.json`
- `configs/scenarios/dense_tasks.json`

## Notes
- Dynamic uncertainty/replanning events are left as post-MVP extensions.
- Current implementation focuses on deterministic educational MAPF coordination without heavy CBS-style solvers.
