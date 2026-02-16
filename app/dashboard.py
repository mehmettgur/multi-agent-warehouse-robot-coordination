from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from warehouse_sim.loader import load_scenario
from warehouse_sim.models import (
    AllocationPolicy,
    Mode,
    PlannerAlgorithm,
    PlannerConfig,
    RobotSpec,
    RunResult,
    SimulationConfig,
    TaskSpec,
)
from warehouse_sim.simulator import run_simulation

RESULTS_DIR = ROOT / "results"
SCENARIO_DIR = ROOT / "configs" / "scenarios"
ROBOT_COLORS = [
    "#0ea5e9",
    "#f97316",
    "#10b981",
    "#ef4444",
    "#8b5cf6",
    "#eab308",
    "#14b8a6",
    "#fb7185",
]

METRIC_DOCS = {
    "makespan": "Tum gorevlerin tamamlanmasi icin gecen toplam tick. Dusuk olmasi daha iyi.",
    "total_path_length": "Robotlarin toplam MOVE adim sayisi.",
    "avg_task_completion_time": "Gorevlerin release->completion ortalama suresi.",
    "wait_count": "Toplam WAIT adimi.",
    "collision_count": "Cakisna olay sayisi. Coordinated hedefi 0.",
    "throughput": "Birim zamanda tamamlanan gorev sayisi.",
    "fairness_task_std": "Robotlar arasi gorev dagiliminin std sapmasi.",
    "planner_expanded_nodes_total": "Planner tarafinda genisletilen toplam dugum.",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .sim-panel {
                border: 1px solid #dbe4ef;
                border-radius: 14px;
                padding: 0.9rem;
                background: linear-gradient(180deg, #f8fbff 0%, #f2f7ff 100%);
                margin-bottom: 0.8rem;
            }
            .sim-grid-wrap {
                overflow-x: auto;
                padding: 0.6rem;
                border: 1px solid #dce5f2;
                border-radius: 12px;
                background: #f8fafc;
            }
            .sim-grid {
                display: grid;
                gap: 4px;
                width: max-content;
            }
            .sim-cell {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                font-weight: 700;
                border: 1px solid #d9e3ef;
                color: #334155;
                background: #f8fafc;
                position: relative;
                overflow: hidden;
            }
            .sim-cell.obstacle { background: #374151; border-color: #374151; color: #f8fafc; }
            .sim-cell.pickup { background: #ecfeff; border-color: #22d3ee; color: #155e75; }
            .sim-cell.dropoff { background: #fff7ed; border-color: #fb923c; color: #9a3412; }
            .sim-cell.trail-1 { box-shadow: inset 0 0 0 999px rgba(59, 130, 246, 0.09); }
            .sim-cell.trail-2 { box-shadow: inset 0 0 0 999px rgba(59, 130, 246, 0.16); }
            .sim-cell.trail-3 { box-shadow: inset 0 0 0 999px rgba(59, 130, 246, 0.24); }
            .robot-chip {
                width: 100%;
                height: 100%;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.12);
                font-size: 10px;
                letter-spacing: 0.2px;
            }
            .legend {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin: 0.55rem 0 0.2rem 0;
            }
            .legend-item {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 8px;
                border: 1px solid #dbe4ef;
                border-radius: 999px;
                background: #ffffff;
                font-size: 12px;
            }
            .legend-swatch {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_scenario_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_data(data: dict) -> SimulationConfig:
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_ratio(completed: int, total: int) -> str:
    if total == 0:
        return "0/0"
    percent = (completed / total) * 100
    return f"{completed}/{total} (%{percent:.1f})"


def _render_config_overview(
    config: SimulationConfig,
    run_type: str,
    mode: Mode,
    planner: PlannerConfig,
    allocator: AllocationPolicy,
) -> None:
    st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
    st.markdown("### Senaryo Ozeti")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Senaryo", config.name)
    c2.metric("Grid", f"{config.width}x{config.height}")
    c3.metric("Robot", str(len(config.robots)))
    c4.metric("Gorev", str(len(config.tasks)))
    c5.metric("Max Tick", str(config.max_ticks))
    st.caption(
        " | ".join(
            [
                f"Calisma tipi: {run_type}",
                f"Mod: {mode if run_type == 'Tek Kosum' else 'baseline + coordinated'}",
                f"Planner: {planner.algorithm}",
                f"w={planner.heuristic_weight}",
                f"Allocator: {allocator}",
                f"Seed: {config.seed}",
            ]
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_metric_cards(metrics: dict, title: str) -> None:
    st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Tamamlanan Gorev",
        _format_ratio(metrics["completed_tasks"], metrics["total_tasks"]),
    )
    c2.metric("Cakisma", str(metrics["collision_count"]))
    c3.metric("Makespan", str(metrics["makespan"]))
    c4.metric("Throughput", str(metrics.get("throughput", 0.0)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Toplam Yol", str(metrics["total_path_length"]))
    c6.metric("Ort. Gorev Suresi", str(metrics["avg_task_completion_time"]))
    c7.metric("Bekleme", str(metrics["wait_count"]))
    c8.metric("Fairness STD", str(metrics.get("fairness_task_std", 0.0)))

    st.bar_chart(
        {
            "makespan": metrics["makespan"],
            "path": metrics["total_path_length"],
            "wait": metrics["wait_count"],
            "expanded": metrics.get("planner_expanded_nodes_total", 0),
        },
        height=170,
    )


def _render_comparison_cards(baseline: dict, coordinated: dict) -> None:
    st.markdown("### Kiyas Ozeti (Coordinated - Baseline)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Tamamlanan Gorev",
        _format_ratio(coordinated["completed_tasks"], coordinated["total_tasks"]),
        delta=f"{coordinated['completed_tasks'] - baseline['completed_tasks']:+}",
    )
    c2.metric(
        "Cakisma",
        str(coordinated["collision_count"]),
        delta=f"{coordinated['collision_count'] - baseline['collision_count']:+}",
        delta_color="inverse",
    )
    c3.metric(
        "Makespan",
        str(coordinated["makespan"]),
        delta=f"{coordinated['makespan'] - baseline['makespan']:+}",
        delta_color="inverse",
    )
    c4.metric(
        "Throughput",
        str(coordinated.get("throughput", 0.0)),
        delta=f"{coordinated.get('throughput', 0.0) - baseline.get('throughput', 0.0):+.4f}",
    )


def _build_trail_map(timeline: list, tick_index: int, window: int = 6) -> dict[tuple[int, int], int]:
    start = max(0, tick_index - window)
    trail: dict[tuple[int, int], int] = {}
    for idx in range(start, tick_index + 1):
        snapshot = timeline[idx].to_dict()
        for pos in snapshot["robot_positions"].values():
            cell = (pos[0], pos[1])
            trail[cell] = trail.get(cell, 0) + 1
    return trail


def _parse_heatmap(metrics: dict) -> dict[tuple[int, int], int]:
    heat: dict[tuple[int, int], int] = {}
    for key, value in metrics.get("congestion_heatmap", {}).items():
        try:
            x_str, y_str = key.split(",")
            heat[(int(x_str), int(y_str))] = int(value)
        except ValueError:
            continue
    return heat


def _cell_base_class(config: SimulationConfig, x: int, y: int) -> str:
    if (x, y) in config.obstacles:
        return "obstacle"
    if (x, y) in set(config.stations.get("pickups", [])):
        return "pickup"
    if (x, y) in set(config.stations.get("dropoffs", [])):
        return "dropoff"
    return ""


def _heat_style(cell: tuple[int, int], heatmap: dict[tuple[int, int], int]) -> str:
    if cell not in heatmap:
        return ""
    max_count = max(heatmap.values()) if heatmap else 1
    alpha = min(0.45, (heatmap[cell] / max_count) * 0.45)
    return f"box-shadow: inset 0 0 0 999px rgba(239, 68, 68, {alpha});"


def _render_grid_html(
    config: SimulationConfig,
    snapshot: dict,
    robot_color_map: dict[str, str],
    trail_map: dict[tuple[int, int], int],
    heatmap: dict[tuple[int, int], int],
    show_heatmap: bool,
) -> str:
    position_to_robot = {
        (pos[0], pos[1]): rid for rid, pos in snapshot["robot_positions"].items()
    }
    blocked_now = {tuple(cell) for cell in snapshot.get("blocked_cells", [])}

    cells: list[str] = []
    for y in range(config.height):
        for x in range(config.width):
            cls = _cell_base_class(config, x, y)
            if (x, y) in blocked_now:
                cls = "obstacle"
            trail_level = min(3, trail_map.get((x, y), 0))
            trail_cls = f"trail-{trail_level}" if trail_level > 0 and cls != "obstacle" else ""
            class_name = " ".join(part for part in ["sim-cell", cls, trail_cls] if part)

            heat_style = _heat_style((x, y), heatmap) if show_heatmap and cls != "obstacle" else ""
            style_attr = f" style='{heat_style}'" if heat_style else ""

            robot_id = position_to_robot.get((x, y))
            if robot_id is not None:
                color = robot_color_map[robot_id]
                label = escape(robot_id)
                content = f"<span class='robot-chip' style='background:{color};'>{label}</span>"
            elif cls == "pickup":
                content = "P"
            elif cls == "dropoff":
                content = "D"
            elif cls == "obstacle":
                content = "#"
            else:
                content = ""

            cells.append(f"<div class='{class_name}'{style_attr}>{content}</div>")

    return (
        f"<div class='sim-grid-wrap'><div class='sim-grid' "
        f"style='grid-template-columns: repeat({config.width}, 32px);'>{''.join(cells)}</div></div>"
    )


def _render_robot_legend(robot_ids: list[str], robot_color_map: dict[str, str]) -> None:
    parts = ["<div class='legend'>"]
    for rid in robot_ids:
        color = robot_color_map[rid]
        parts.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color};'></span>{escape(rid)}"
            "</span>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _replay_controls(key_prefix: str, timeline_len: int, step_size: int) -> int:
    max_tick = timeline_len - 1
    state_key = f"{key_prefix}_tick"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    c1, c2, c3, c4, c5 = st.columns([1, 1, 4, 1, 1])
    if c1.button("|<", key=f"{key_prefix}_first"):
        st.session_state[state_key] = 0
    if c2.button("<", key=f"{key_prefix}_prev"):
        st.session_state[state_key] = max(0, st.session_state[state_key] - step_size)
    if c4.button(">", key=f"{key_prefix}_next"):
        st.session_state[state_key] = min(max_tick, st.session_state[state_key] + step_size)
    if c5.button(">|", key=f"{key_prefix}_last"):
        st.session_state[state_key] = max_tick

    slider_tick = c3.slider(
        "Tick",
        min_value=0,
        max_value=max_tick,
        value=int(st.session_state[state_key]),
        key=f"{key_prefix}_slider",
        label_visibility="collapsed",
    )
    st.session_state[state_key] = slider_tick
    st.progress((slider_tick + 1) / timeline_len)
    return slider_tick


def _render_replay(
    result: RunResult,
    config: SimulationConfig,
    key_prefix: str,
    title: str,
    show_heatmap: bool,
    step_size: int,
) -> None:
    st.markdown(f"### {title}")
    if not result.timeline:
        st.info("Timeline olusmadi.")
        return

    tick_index = _replay_controls(
        key_prefix=key_prefix,
        timeline_len=len(result.timeline),
        step_size=step_size,
    )
    snapshot = result.timeline[tick_index].to_dict()

    robot_ids = sorted(snapshot["robot_positions"].keys())
    robot_color_map = {
        rid: ROBOT_COLORS[idx % len(ROBOT_COLORS)] for idx, rid in enumerate(robot_ids)
    }

    trail_map = _build_trail_map(result.timeline, tick_index=tick_index)
    heatmap = _parse_heatmap(result.metrics.to_dict())

    st.caption(
        f"Tick {snapshot['tick']} | Tamamlanan Gorev: {snapshot['completed_tasks']} | "
        f"Bu tick cakisna: {snapshot['collision_events']}"
    )
    st.markdown(
        _render_grid_html(
            config=config,
            snapshot=snapshot,
            robot_color_map=robot_color_map,
            trail_map=trail_map,
            heatmap=heatmap,
            show_heatmap=show_heatmap,
        ),
        unsafe_allow_html=True,
    )
    _render_robot_legend(robot_ids, robot_color_map)


def _render_metric_guide() -> None:
    with st.expander("Metrikler ne anlama geliyor?", expanded=False):
        st.markdown("\n".join([f"- `{k}`: {v}" for k, v in METRIC_DOCS.items()]))


def _save_result_files(prefix: str, payload: dict, rows: list[dict] | None = None) -> list[str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{prefix}_{timestamp}.json"
    _write_json(json_path, payload)
    saved = [str(json_path)]

    if rows:
        csv_path = RESULTS_DIR / f"{prefix}_{timestamp}.csv"
        _write_csv(csv_path, rows)
        saved.append(str(csv_path))

    return saved


def _run_ablation(
    scenario_paths: list[str],
    seed_override: int | None,
    planners: list[PlannerConfig],
    allocators: list[AllocationPolicy],
    modes: list[Mode],
) -> list[dict]:
    rows: list[dict] = []

    for scenario_path in scenario_paths:
        config = load_scenario(scenario_path)
        use_seed = config.seed if seed_override is None else seed_override

        for planner in planners:
            for allocator in allocators:
                for mode in modes:
                    result = run_simulation(
                        config=config,
                        mode=mode,
                        seed=use_seed,
                        planner_override=planner,
                        allocator_override=allocator,
                    )
                    rows.append(
                        {
                            "scenario": config.name,
                            "scenario_file": Path(scenario_path).name,
                            "seed": use_seed,
                            "mode": mode,
                            "planner": planner.algorithm,
                            "heuristic_weight": planner.heuristic_weight,
                            "allocator": allocator,
                            **result.metrics.to_dict(),
                        }
                    )
    return rows


def _show_editor(selected_path: Path) -> None:
    st.subheader("Scenario Editor (Opsiyonel)")
    st.caption("Sadece ileri duzey ihtiyaclar icin.")

    with st.expander("JSON Duzenleyici", expanded=False):
        st.text_area(
            "Scenario JSON",
            key="scenario_json_text",
            height=420,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("JSON Validate"):
                try:
                    parsed = json.loads(st.session_state["scenario_json_text"])
                    preview = _config_from_data(parsed)
                    st.success(
                        f"Gecerli: {preview.name} | grid={preview.width}x{preview.height} | "
                        f"robots={len(preview.robots)} | tasks={len(preview.tasks)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Gecersiz JSON: {exc}")

        with c2:
            save_name = st.text_input(
                "Dosya adi",
                value=selected_path.name,
                key="save_scenario_name",
            )
            if st.button("Scenario Kaydet"):
                try:
                    parsed = json.loads(st.session_state["scenario_json_text"])
                    target_name = save_name if save_name.endswith(".json") else f"{save_name}.json"
                    target = SCENARIO_DIR / Path(target_name).name
                    target.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                    st.success(f"Kaydedildi: {target}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Kaydetme hatasi: {exc}")


def main() -> None:
    st.set_page_config(page_title="Warehouse Robot Coordination", layout="wide")
    _inject_styles()

    st.title("Multi-Agent Depo Robot Koordinasyonu")
    st.caption("Planner/allocator ablation + gelismis replay + heatmap overlay")

    scenario_paths = sorted(SCENARIO_DIR.glob("*.json"))
    if not scenario_paths:
        st.error("configs/scenarios altinda scenario dosyasi bulunamadi.")
        return

    st.sidebar.subheader("Calistirma Ayarlari")
    selected_path_str = st.sidebar.selectbox("Scenario", [str(path) for path in scenario_paths])
    selected_path = Path(selected_path_str)
    base_data = _load_scenario_data(selected_path)

    if st.session_state.get("selected_scenario") != str(selected_path):
        st.session_state["selected_scenario"] = str(selected_path)
        st.session_state["scenario_json_text"] = json.dumps(base_data, indent=2)

    run_type = st.sidebar.radio("Run Type", ["Tek Kosum", "Baseline vs Coordinated"], index=1)
    mode: Mode = "coordinated"
    if run_type == "Tek Kosum":
        mode = st.sidebar.selectbox("Mode", ["coordinated", "baseline"])

    planner_algorithm: PlannerAlgorithm = st.sidebar.selectbox(
        "Planner",
        ["astar", "dijkstra", "weighted_astar"],
    )
    heuristic_weight = 1.0
    if planner_algorithm == "weighted_astar":
        heuristic_weight = st.sidebar.slider("Weighted A* w", 1.0, 3.0, 1.4, 0.1)

    allocator_policy: AllocationPolicy = st.sidebar.selectbox(
        "Allocator",
        ["hungarian", "greedy"],
    )

    seed = int(st.sidebar.number_input("Seed", min_value=0, value=int(base_data.get("seed", 0)), step=1))
    max_ticks = int(
        st.sidebar.number_input(
            "Max Ticks",
            min_value=1,
            value=int(base_data.get("simulation", {}).get("max_ticks", 200)),
            step=1,
        )
    )

    playback_speed = int(st.sidebar.slider("Playback speed", min_value=1, max_value=5, value=2, step=1))
    show_heatmap = st.sidebar.checkbox("Heatmap overlay", value=True)
    use_editor = st.sidebar.checkbox("Editor JSON'u kullan", value=False)
    persist_results = st.sidebar.checkbox("Sonuclari results/ altina kaydet", value=True)
    run_clicked = st.sidebar.button("Simulasyonu Baslat", type="primary")

    planner_cfg = PlannerConfig(
        algorithm=planner_algorithm,
        heuristic_weight=heuristic_weight,
    )

    if run_clicked:
        try:
            if use_editor:
                scenario_data = json.loads(st.session_state["scenario_json_text"])
                config = _config_from_data(scenario_data)
            else:
                config = load_scenario(selected_path)
            config = replace(
                config,
                seed=seed,
                max_ticks=max_ticks,
                planner=planner_cfg,
                allocator_policy=allocator_policy,
            )

            payload: dict = {
                "config": config,
                "run_type": run_type,
                "mode": mode,
                "planner": planner_cfg,
                "allocator": allocator_policy,
                "single": None,
                "baseline": None,
                "coordinated": None,
                "saved_files": [],
            }

            if run_type == "Tek Kosum":
                result = run_simulation(
                    config=config,
                    mode=mode,
                    seed=config.seed,
                    planner_override=planner_cfg,
                    allocator_override=allocator_policy,
                )
                payload["single"] = result
                if persist_results:
                    saved = _save_result_files(
                        prefix=f"single_{config.name.lower().replace(' ', '_')}",
                        payload=result.to_dict(),
                    )
                    payload["saved_files"] = saved
            else:
                baseline = run_simulation(
                    config=config,
                    mode="baseline",
                    seed=config.seed,
                    planner_override=planner_cfg,
                    allocator_override=allocator_policy,
                )
                coordinated = run_simulation(
                    config=config,
                    mode="coordinated",
                    seed=config.seed,
                    planner_override=planner_cfg,
                    allocator_override=allocator_policy,
                )
                payload["baseline"] = baseline
                payload["coordinated"] = coordinated

                if persist_results:
                    rows = [
                        {
                            "mode": "baseline",
                            "planner": planner_cfg.algorithm,
                            "heuristic_weight": planner_cfg.heuristic_weight,
                            "allocator": allocator_policy,
                            **baseline.metrics.to_dict(),
                        },
                        {
                            "mode": "coordinated",
                            "planner": planner_cfg.algorithm,
                            "heuristic_weight": planner_cfg.heuristic_weight,
                            "allocator": allocator_policy,
                            **coordinated.metrics.to_dict(),
                        },
                    ]
                    saved = _save_result_files(
                        prefix=f"comparison_{config.name.lower().replace(' ', '_')}",
                        payload={
                            "scenario": config.name,
                            "seed": config.seed,
                            "planner": planner_cfg.algorithm,
                            "heuristic_weight": planner_cfg.heuristic_weight,
                            "allocator": allocator_policy,
                            "baseline": baseline.metrics.to_dict(),
                            "coordinated": coordinated.metrics.to_dict(),
                        },
                        rows=rows,
                    )
                    payload["saved_files"] = saved

            st.session_state["last_run_payload"] = payload
        except Exception as exc:  # noqa: BLE001
            st.error(f"Calistirma hatasi: {exc}")

    sim_tab, ablation_tab, editor_tab = st.tabs(["Simulation", "Ablation", "Scenario Editor"])

    with sim_tab:
        payload = st.session_state.get("last_run_payload")
        if payload is None:
            st.info("Soldan ayarlari secip 'Simulasyonu Baslat' butonuna tikla.")
        else:
            config: SimulationConfig = payload["config"]
            run_type_payload = payload["run_type"]
            mode_payload: Mode = payload["mode"]
            planner_payload: PlannerConfig = payload["planner"]
            allocator_payload: AllocationPolicy = payload["allocator"]

            _render_config_overview(
                config=config,
                run_type=run_type_payload,
                mode=mode_payload,
                planner=planner_payload,
                allocator=allocator_payload,
            )
            _render_metric_guide()

            if run_type_payload == "Tek Kosum":
                result: RunResult = payload["single"]
                _render_metric_cards(result.metrics.to_dict(), f"Metrikler ({mode_payload})")
                _render_replay(
                    result=result,
                    config=config,
                    key_prefix=f"single_{mode_payload}",
                    title="Robot Hareket Replay",
                    show_heatmap=show_heatmap,
                    step_size=playback_speed,
                )
            else:
                baseline: RunResult = payload["baseline"]
                coordinated: RunResult = payload["coordinated"]

                _render_comparison_cards(
                    baseline=baseline.metrics.to_dict(),
                    coordinated=coordinated.metrics.to_dict(),
                )

                c1, c2 = st.columns(2)
                with c1:
                    _render_metric_cards(baseline.metrics.to_dict(), "Baseline Metrikleri")
                    _render_replay(
                        result=baseline,
                        config=config,
                        key_prefix="baseline",
                        title="Baseline Replay",
                        show_heatmap=show_heatmap,
                        step_size=playback_speed,
                    )
                with c2:
                    _render_metric_cards(coordinated.metrics.to_dict(), "Coordinated Metrikleri")
                    _render_replay(
                        result=coordinated,
                        config=config,
                        key_prefix="coordinated",
                        title="Coordinated Replay",
                        show_heatmap=show_heatmap,
                        step_size=playback_speed,
                    )

            for path in payload.get("saved_files", []):
                st.caption(f"Kaydedildi: {path}")

    with ablation_tab:
        st.subheader("Ablation Study")
        st.caption("scenario x mode x planner x allocator kombinasyonlarini calistirir.")

        ab_scenarios = st.multiselect(
            "Scenarios",
            [str(path) for path in scenario_paths],
            default=[str(selected_path)],
        )
        ab_planners = st.multiselect(
            "Planners",
            ["astar", "dijkstra", "weighted_astar"],
            default=["astar", "dijkstra", "weighted_astar"],
        )
        ab_allocators = st.multiselect(
            "Allocators",
            ["greedy", "hungarian"],
            default=["greedy", "hungarian"],
        )
        ab_modes = st.multiselect(
            "Modes",
            ["baseline", "coordinated"],
            default=["baseline", "coordinated"],
        )

        ab_weight = st.slider("Weighted A* w (ablation)", 1.0, 3.0, 1.4, 0.1)
        run_ablation_clicked = st.button("Ablation Calistir")

        if run_ablation_clicked:
            planners: list[PlannerConfig] = []
            for name in ab_planners:
                if name == "weighted_astar":
                    planners.append(PlannerConfig(algorithm=name, heuristic_weight=ab_weight))
                else:
                    planners.append(PlannerConfig(algorithm=name, heuristic_weight=1.0))

            rows = _run_ablation(
                scenario_paths=ab_scenarios,
                seed_override=seed,
                planners=planners,
                allocators=ab_allocators,
                modes=ab_modes,
            )
            st.session_state["ablation_rows"] = rows

            saved = _save_result_files(
                prefix="ablation_ui",
                payload={"rows": rows, "num_rows": len(rows)},
                rows=rows,
            )
            st.session_state["ablation_saved"] = saved

        rows = st.session_state.get("ablation_rows")
        if rows:
            st.dataframe(rows, width="stretch")
            st.bar_chart(
                {
                    "avg_makespan": sum(float(r["makespan"]) for r in rows) / max(1, len(rows)),
                    "avg_collision": sum(float(r["collision_count"]) for r in rows) / max(1, len(rows)),
                    "avg_throughput": sum(float(r.get("throughput", 0.0)) for r in rows) / max(1, len(rows)),
                },
                height=190,
            )

            for path in st.session_state.get("ablation_saved", []):
                st.caption(f"Ablation kaydi: {path}")

    with editor_tab:
        _show_editor(selected_path)


if __name__ == "__main__":
    main()
