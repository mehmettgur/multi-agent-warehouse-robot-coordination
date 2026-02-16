from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from warehouse_sim.loader import load_scenario
from warehouse_sim.runner import run_comparison
from warehouse_sim.simulator import run_simulation


def _render_grid(snapshot: dict, config) -> str:
    cells = [["." for _ in range(config.width)] for _ in range(config.height)]

    for ox, oy in config.obstacles:
        cells[oy][ox] = "#"

    for pos in config.stations.get("pickups", []):
        x, y = pos
        if cells[y][x] == ".":
            cells[y][x] = "P"

    for pos in config.stations.get("dropoffs", []):
        x, y = pos
        if cells[y][x] == ".":
            cells[y][x] = "D"

    for robot_id, pos in snapshot["robot_positions"].items():
        x, y = pos
        label = robot_id[-1].upper() if robot_id else "R"
        cells[y][x] = label

    return "\n".join(" ".join(row) for row in cells)


def main() -> None:
    st.set_page_config(page_title="Warehouse Robot Coordination", layout="wide")
    st.title("Multi-Agent Depo Robot Koordinasyonu")

    scenario_paths = sorted(Path("configs/scenarios").glob("*.json"))
    if not scenario_paths:
        st.error("No scenario files found in configs/scenarios")
        return

    scenario_str = st.sidebar.selectbox(
        "Scenario",
        [str(path) for path in scenario_paths],
    )
    mode = st.sidebar.selectbox("Mode", ["coordinated", "baseline"])
    seed = st.sidebar.number_input("Seed", min_value=0, value=0, step=1)
    compare = st.sidebar.checkbox("Compare baseline vs coordinated", value=True)

    if st.sidebar.button("Run Simulation"):
        scenario_path = str(scenario_str)
        config = load_scenario(scenario_path)

        st.subheader("Configuration")
        st.code(
            json.dumps(
                {
                    "name": config.name,
                    "grid": {"width": config.width, "height": config.height},
                    "robots": len(config.robots),
                    "tasks": len(config.tasks),
                    "max_ticks": config.max_ticks,
                },
                indent=2,
            ),
            language="json",
        )

        if compare:
            payload = run_comparison(
                scenario_path=scenario_path,
                seed=int(seed),
                output_dir="results",
            )
            st.subheader("Comparison")
            st.dataframe(
                [
                    {"mode": "baseline", **payload["baseline"]},
                    {"mode": "coordinated", **payload["coordinated"]},
                ],
                use_container_width=True,
            )

        result = run_simulation(config=config, mode=mode, seed=int(seed))

        st.subheader("Run Metrics")
        st.json(result.metrics.to_dict())

        st.subheader("Grid Replay")
        if not result.timeline:
            st.info("No timeline generated.")
            return

        tick = st.slider(
            "Tick",
            min_value=0,
            max_value=len(result.timeline) - 1,
            value=0,
            step=1,
        )
        snapshot = result.timeline[tick].to_dict()
        st.caption(f"Tick {snapshot['tick']}")
        st.code(_render_grid(snapshot, config), language="text")


if __name__ == "__main__":
    main()
