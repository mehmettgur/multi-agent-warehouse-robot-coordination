from __future__ import annotations

import csv
import json
from dataclasses import replace
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
    CoordinationConfig,
    Mode,
    PlannerAlgorithm,
    PlannerConfig,
    RunResult,
    SimulationConfig,
)
from warehouse_sim.paper_tables import ALLOCATOR_LABELS, MODE_LABELS, PLANNER_LABELS
from warehouse_sim.runner import DEFAULT_ROBUSTNESS_SEEDS, run_suite
from warehouse_sim.scenario_catalog import (
    APPENDIX_SCENARIOS,
    CORE_SCENARIOS,
    LEGACY_SCENARIOS,
    scenario_path,
)
from warehouse_sim.simulator import run_simulation

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
SUITE_LABELS = {
    "main": "Ana Karşılaştırma",
    "allocator": "Atayıcı Ablation",
    "planner": "Planlayıcı Ablation",
    "coordination": "Koordinasyon Ablation",
    "robustness": "Robustness / Appendix",
    "all": "Tam Paper Pack",
}
PRIMARY_METRICS = [
    "Tamamlama",
    "Çakışma",
    "Makespan",
    "Throughput",
]
SECONDARY_METRICS = [
    "Toplam Yol",
    "Bekleme",
    "Ort. Görev Süresi",
    "Adillik Std",
]
METRIC_EXPLANATIONS = {
    "Tamamlama": "Görevlerin kaçının tamamlandığını gösterir. Yüksek olması daha iyidir.",
    "Çakışma": "Aynı hücreye girme veya swap çatışması sayısıdır. Koordineli mod hedefi 0'dır.",
    "Makespan": "Tüm görevlerin bitmesi için geçen toplam tik sayısıdır. Düşük olması daha iyidir.",
    "Throughput": "Birim zamanda tamamlanan görev sayısıdır. Yüksek olması daha iyidir.",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
            .hero {
                border: 1px solid #dbe4ef;
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
                margin-bottom: 1rem;
            }
            .hero h1 {
                font-size: 1.9rem;
                margin: 0 0 0.25rem 0;
                color: #0f172a;
            }
            .hero p {
                margin: 0;
                color: #475569;
                font-size: 0.98rem;
            }
            .metric-note {
                border: 1px solid #dbe4ef;
                background: #f8fafc;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin: 0.8rem 0 1rem 0;
            }
            .metric-note strong { color: #0f172a; }
            .sim-panel {
                border: 1px solid #dbe4ef;
                border-radius: 16px;
                padding: 0.95rem;
                background: #ffffff;
                margin-bottom: 0.85rem;
            }
            .sim-grid-wrap {
                overflow-x: auto;
                padding: 0.65rem;
                border: 1px solid #dce5f2;
                border-radius: 14px;
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
            .sim-cell.trail-1 { box-shadow: inset 0 0 0 999px rgba(59, 130, 246, 0.08); }
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


def _scenario_names(include_appendix: bool, include_legacy: bool) -> list[str]:
    names = list(CORE_SCENARIOS)
    if include_appendix:
        names.extend(APPENDIX_SCENARIOS)
    if include_legacy:
        names.extend(LEGACY_SCENARIOS)
    return names


def _planner_label(algorithm: PlannerAlgorithm) -> str:
    return PLANNER_LABELS.get(algorithm, algorithm)


def _allocator_label(policy: AllocationPolicy) -> str:
    return ALLOCATOR_LABELS.get(policy, policy)


def _mode_label(mode: Mode) -> str:
    return MODE_LABELS.get(mode, mode)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _format_completion(metrics: dict) -> str:
    completed = int(metrics.get("completed_tasks", 0))
    total = int(metrics.get("total_tasks", 0))
    if total <= 0:
        return "0/0"
    return f"{completed}/{total}"


def _metric_value(label: str, metrics: dict) -> str:
    if label == "Tamamlama":
        return _format_completion(metrics)
    value = metrics.get({
        "Çakışma": "collision_count",
        "Makespan": "makespan",
        "Throughput": "throughput",
        "Toplam Yol": "total_path_length",
        "Bekleme": "wait_count",
        "Ort. Görev Süresi": "avg_task_completion_time",
        "Adillik Std": "fairness_task_std",
    }[label], 0)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _render_metric_explanations() -> None:
    parts = ["<div class='metric-note'><strong>Ana metrikler</strong><br>"]
    for title, text in METRIC_EXPLANATIONS.items():
        parts.append(f"<div><strong>{escape(title)}:</strong> {escape(text)}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_run_overview(config: SimulationConfig, planner: PlannerConfig, allocator: AllocationPolicy, run_mode: str) -> None:
    st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Senaryo", config.name)
    c2.metric("Grid", f"{config.width}x{config.height}")
    c3.metric("Robot", str(len(config.robots)))
    c4.metric("Görev", str(len(config.tasks)))
    c5.metric("Maks. Tick", str(config.max_ticks))
    st.caption(
        " | ".join(
            [
                f"Akış: {run_mode}",
                f"Planlayıcı: {_planner_label(planner.algorithm)}",
                f"w={planner.heuristic_weight:.1f}",
                f"Atayıcı: {_allocator_label(allocator)}",
                f"Seed: {config.seed}",
            ]
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_primary_cards(metrics: dict, title: str) -> None:
    st.markdown(f"#### {title}")
    cols = st.columns(4)
    for col, label in zip(cols, PRIMARY_METRICS):
        col.metric(label, _metric_value(label, metrics))


def _render_secondary_cards(metrics: dict) -> None:
    cols = st.columns(4)
    for col, label in zip(cols, SECONDARY_METRICS):
        col.metric(label, _metric_value(label, metrics))


def _render_comparison_cards(baseline: dict, coordinated: dict) -> None:
    st.markdown("### Kıyas Özeti")
    cols = st.columns(4)
    for idx, label in enumerate(PRIMARY_METRICS):
        baseline_value = _metric_value(label, baseline)
        coordinated_value = _metric_value(label, coordinated)
        delta = None
        if label == "Tamamlama":
            delta = f"{int(coordinated.get('completed_tasks', 0)) - int(baseline.get('completed_tasks', 0)):+}"
        elif label == "Çakışma":
            delta = f"{int(coordinated.get('collision_count', 0)) - int(baseline.get('collision_count', 0)):+}"
        elif label == "Makespan":
            delta = f"{int(coordinated.get('makespan', 0)) - int(baseline.get('makespan', 0)):+}"
        elif label == "Throughput":
            delta = f"{float(coordinated.get('throughput', 0.0)) - float(baseline.get('throughput', 0.0)):+.4f}"
        cols[idx].metric(label, coordinated_value, delta=delta)
    st.caption("Kart değerleri koordineli modu gösterir; delta sütunu baseline farkını verir.")


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
    alpha = min(0.42, (heatmap[cell] / max_count) * 0.42)
    return f"box-shadow: inset 0 0 0 999px rgba(239, 68, 68, {alpha});"


def _render_grid_html(
    config: SimulationConfig,
    snapshot: dict,
    robot_color_map: dict[str, str],
    trail_map: dict[tuple[int, int], int],
    heatmap: dict[tuple[int, int], int],
    show_heatmap: bool,
) -> str:
    position_to_robot = {(pos[0], pos[1]): rid for rid, pos in snapshot["robot_positions"].items()}
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
                content = (
                    f"<span class='robot-chip' style='background:{robot_color_map[robot_id]};'>"
                    f"{escape(robot_id)}</span>"
                )
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
        f"<div class='sim-grid-wrap'><div class='sim-grid' style='grid-template-columns: repeat({config.width}, 32px);'>"
        f"{''.join(cells)}</div></div>"
    )


def _render_robot_legend(robot_ids: list[str], robot_color_map: dict[str, str]) -> None:
    parts = ["<div class='legend'>"]
    for rid in robot_ids:
        parts.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{robot_color_map[rid]};'></span>{escape(rid)}"
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


def _render_replay(result: RunResult, config: SimulationConfig, key_prefix: str, title: str, show_heatmap: bool, step_size: int) -> None:
    st.markdown(f"#### {title}")
    if not result.timeline:
        st.info("Zaman çizelgesi oluşmadı.")
        return

    tick_index = _replay_controls(key_prefix=key_prefix, timeline_len=len(result.timeline), step_size=step_size)
    snapshot = result.timeline[tick_index].to_dict()
    robot_ids = sorted(snapshot["robot_positions"].keys())
    robot_color_map = {rid: ROBOT_COLORS[idx % len(ROBOT_COLORS)] for idx, rid in enumerate(robot_ids)}
    trail_map = _build_trail_map(result.timeline, tick_index=tick_index)
    heatmap = _parse_heatmap(result.metrics.to_dict())

    st.caption(
        f"Tick {snapshot['tick']} | Tamamlanan görev: {snapshot['completed_tasks']} | Bu tikte çakışma: {snapshot['collision_events']}"
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


def _render_table(title: str, path: str | None) -> None:
    if not path:
        return
    rows = _read_csv_rows(path)
    st.markdown(f"### {title}")
    st.dataframe(rows, width="stretch")


def _render_svg_preview(title: str, path: str | None) -> None:
    if not path:
        return
    svg_path = Path(path)
    st.markdown(f"### {title}")
    if svg_path.suffix.lower() == ".svg":
        st.image(str(svg_path), width="stretch")
    else:
        st.image(str(svg_path), width="stretch")
    st.caption(str(svg_path))


def _resolve_table_path(payload: dict, suite_key: str, preferred_suffix: str = ".csv") -> str | None:
    files = payload.get("tables", {}).get(suite_key, [])
    for file_path in files:
        if file_path.endswith(preferred_suffix):
            return file_path
    return None


def _resolve_figure_path(payload: dict, figure_key: str) -> str | None:
    files = payload.get("figures", {}).get(figure_key, [])
    for file_path in files:
        if file_path.endswith(".svg") or file_path.endswith(".png"):
            return file_path
    return None


def _coordination_for_mode(mode: Mode) -> CoordinationConfig:
    if mode == "baseline":
        return CoordinationConfig(variant="independent")
    if mode == "baseline_priority":
        return CoordinationConfig(
            variant="priority_static",
            edge_conflicts=True,
            dynamic_priority=False,
            micro_replan=False,
        )
    return CoordinationConfig(
        variant="full",
        edge_conflicts=True,
        dynamic_priority=True,
        micro_replan=True,
    )


def _run_demo(mode: Mode | None, scenario_name: str, planner: PlannerConfig, allocator: AllocationPolicy, seed: int, max_ticks: int) -> dict:
    config = replace(
        load_scenario(scenario_path(scenario_name)),
        seed=seed,
        max_ticks=max_ticks,
        planner=planner,
        allocator_policy=allocator,
    )
    if mode is None:
        baseline_coordination = _coordination_for_mode("baseline")
        coordinated_coordination = _coordination_for_mode("coordinated")
        baseline = run_simulation(
            config=config,
            mode="baseline",
            seed=config.seed,
            planner_override=planner,
            allocator_override=allocator,
            coordination_override=baseline_coordination,
        )
        coordinated = run_simulation(
            config=config,
            mode="coordinated",
            seed=config.seed,
            planner_override=planner,
            allocator_override=allocator,
            coordination_override=coordinated_coordination,
        )
        return {"config": config, "compare": True, "baseline": baseline, "coordinated": coordinated}

    single = run_simulation(
        config=config,
        mode=mode,
        seed=config.seed,
        planner_override=planner,
        allocator_override=allocator,
        coordination_override=_coordination_for_mode(mode),
    )
    return {"config": config, "compare": False, "mode": mode, "single": single}


def _paper_pack_summary(payload: dict) -> None:
    st.success(
        " | ".join(
            [
                f"Suite: {SUITE_LABELS[payload['suite']]}",
                f"Ham CSV: {payload['raw_csv']}",
                f"Satır: {payload['num_rows']}",
            ]
        )
    )


def main() -> None:
    st.set_page_config(page_title="Warehouse Robot Coordination", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Multi-Agent Depo Robot Koordinasyonu</h1>
            <p>Sunum için sade demo akışı ve LaTeX makale için tekrar üretilebilir paper pack çıktıları.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_tab, paper_tab = st.tabs(["Demo", "Paper Pack"])

    with demo_tab:
        st.subheader("Karşılaştırmalı Demo")
        st.caption("Ana anlatı: aynı senaryoda baseline ve koordineli mod farkını hızlı ve temiz şekilde göstermek.")

        with st.expander("Gelişmiş demo seçenekleri", expanded=False):
            st.checkbox("Appendix senaryolarını göster", value=False, key="demo_appendix")
            st.checkbox("Legacy senaryolarını göster", value=False, key="demo_legacy")
            st.checkbox("Tek koşum modunu aç", value=False, key="demo_single_mode")
            st.checkbox("Isı haritasını göster", value=True, key="demo_heatmap")
            st.slider("Replay adım boyu", min_value=1, max_value=5, value=2, key="demo_speed")

        scenario_options = _scenario_names(include_appendix=st.session_state.get("demo_appendix", False), include_legacy=st.session_state.get("demo_legacy", False))
        default_index = scenario_options.index("narrow_corridor_swap") if "narrow_corridor_swap" in scenario_options else 0

        with st.form("demo_form"):
            c1, c2, c3, c4, c5 = st.columns(5)
            scenario_name = c1.selectbox("Senaryo", scenario_options, index=default_index)
            if st.session_state.get("demo_single_mode", False):
                run_mode_label = c2.selectbox(
                    "Akış",
                    [
                        "Karşılaştırma",
                        "Tek Koşum - Koordineli",
                        "Tek Koşum - Baseline",
                        "Tek Koşum - Öncelikli Baseline",
                    ],
                    index=0,
                )
            else:
                run_mode_label = c2.selectbox("Akış", ["Karşılaştırma"], index=0)
            planner_label = c3.selectbox("Planlayıcı", list(PLANNER_LABELS.values()), index=0)
            allocator_label = c4.selectbox("Atayıcı", [_allocator_label("greedy"), _allocator_label("hungarian")], index=0)
            seed = int(c5.number_input("Seed", min_value=0, value=int(load_scenario(scenario_path(scenario_name)).seed), step=1))

            planner_algorithm = next(key for key, value in PLANNER_LABELS.items() if value == planner_label)
            heuristic_weight = 1.0
            if planner_algorithm == "weighted_astar":
                heuristic_weight = st.slider("A* ağırlık (w)", min_value=1.0, max_value=3.0, value=1.4, step=0.1)
            max_ticks = int(st.number_input("Maksimum tick", min_value=1, value=int(load_scenario(scenario_path(scenario_name)).max_ticks), step=1))
            run_demo = st.form_submit_button("Demo Çalıştır", type="primary")

        if run_demo:
            planner = PlannerConfig(algorithm=planner_algorithm, heuristic_weight=heuristic_weight)
            allocator = next(key for key, value in ALLOCATOR_LABELS.items() if value == allocator_label)
            mode: Mode | None = None
            if run_mode_label == "Tek Koşum - Koordineli":
                mode = "coordinated"
            elif run_mode_label == "Tek Koşum - Baseline":
                mode = "baseline"
            elif run_mode_label == "Tek Koşum - Öncelikli Baseline":
                mode = "baseline_priority"
            st.session_state["demo_payload"] = _run_demo(
                mode=mode,
                scenario_name=scenario_name,
                planner=planner,
                allocator=allocator,
                seed=seed,
                max_ticks=max_ticks,
            )

        demo_payload = st.session_state.get("demo_payload")
        if demo_payload is None:
            st.info("Varsayılan akış için `narrow_corridor_swap` senaryosunu karşılaştırmalı modda çalıştır.")
        else:
            config: SimulationConfig = demo_payload["config"]
            planner = config.planner
            allocator = config.allocator_policy
            run_mode = "Karşılaştırma" if demo_payload["compare"] else _mode_label(demo_payload["mode"])
            _render_run_overview(config, planner, allocator, run_mode)
            _render_metric_explanations()

            if demo_payload["compare"]:
                baseline: RunResult = demo_payload["baseline"]
                coordinated: RunResult = demo_payload["coordinated"]
                _render_comparison_cards(baseline.metrics.to_dict(), coordinated.metrics.to_dict())
                with st.expander("İkincil metrikler", expanded=False):
                    left, right = st.columns(2)
                    with left:
                        _render_secondary_cards(baseline.metrics.to_dict())
                    with right:
                        _render_secondary_cards(coordinated.metrics.to_dict())
                left, right = st.columns(2)
                with left:
                    _render_primary_cards(baseline.metrics.to_dict(), "Baseline")
                    _render_replay(
                        baseline,
                        config,
                        key_prefix="demo_baseline",
                        title="Baseline Replay",
                        show_heatmap=st.session_state.get("demo_heatmap", True),
                        step_size=st.session_state.get("demo_speed", 2),
                    )
                with right:
                    _render_primary_cards(coordinated.metrics.to_dict(), "Koordineli")
                    _render_replay(
                        coordinated,
                        config,
                        key_prefix="demo_coordinated",
                        title="Koordineli Replay",
                        show_heatmap=st.session_state.get("demo_heatmap", True),
                        step_size=st.session_state.get("demo_speed", 2),
                    )
            else:
                single: RunResult = demo_payload["single"]
                _render_primary_cards(single.metrics.to_dict(), f"{_mode_label(demo_payload['mode'])} Sonucu")
                _render_secondary_cards(single.metrics.to_dict())
                _render_replay(
                    single,
                    config,
                    key_prefix="demo_single",
                    title="Replay",
                    show_heatmap=st.session_state.get("demo_heatmap", True),
                    step_size=st.session_state.get("demo_speed", 2),
                )

    with paper_tab:
        st.subheader("Paper Pack")
        st.caption("Kanonik benchmark suite'lerini tek komut mantığıyla çalıştırır; sade CSV, LaTeX ve figür çıktıları üretir.")

        with st.form("paper_form"):
            c1, c2, c3 = st.columns([2, 2, 2])
            suite_options = list(SUITE_LABELS.values())
            suite_label = c1.selectbox(
                "Deney paketi",
                suite_options,
                index=suite_options.index(SUITE_LABELS["all"]),
            )
            with_latex = c2.checkbox("LaTeX tablolarını üret", value=True)
            with_figures = c3.checkbox("Figür paketini üret", value=True)
            selected_suite = next(key for key, value in SUITE_LABELS.items() if value == suite_label)
            seeds = DEFAULT_ROBUSTNESS_SEEDS
            if selected_suite in {"robustness", "all"}:
                seeds = st.multiselect(
                    "Robustness seed listesi",
                    options=[11, 17, 23, 31, 37, 41, 53],
                    default=DEFAULT_ROBUSTNESS_SEEDS,
                )
            run_paper = st.form_submit_button("Paper Pack Çalıştır", type="primary")

        if run_paper:
            st.session_state["paper_payload"] = run_suite(
                suite=selected_suite,
                output_dir="results",
                seeds=list(seeds),
                with_latex=with_latex,
                with_figures=with_figures,
            )

        paper_payload = st.session_state.get("paper_payload")
        if paper_payload is None:
            st.info("Varsayılan öneri: `Tam Paper Pack` + LaTeX + figür üretimi.")
        else:
            _paper_pack_summary(paper_payload)

            main_table = _resolve_table_path(paper_payload, "main")
            allocator_table = _resolve_table_path(paper_payload, "allocator")
            planner_table = _resolve_table_path(paper_payload, "planner")
            coordination_table = _resolve_table_path(paper_payload, "coordination")
            robustness_table = _resolve_table_path(paper_payload, "robustness")

            if main_table:
                _render_table("Ana Karşılaştırma Tablosu", main_table)
            if allocator_table:
                _render_table("Atayıcı Ablation Tablosu", allocator_table)
            if planner_table:
                _render_table("Planlayıcı Ablation Tablosu", planner_table)
            if coordination_table:
                _render_table("Koordinasyon Ablation Tablosu", coordination_table)
            if robustness_table:
                with st.expander("Robustness / Appendix Tablosu", expanded=False):
                    _render_table("Robustness Tablosu", robustness_table)

            if paper_payload.get("figures"):
                fig1, fig2, fig3 = st.columns(3)
                with fig1:
                    _render_svg_preview("Swap Demo Figürü", _resolve_figure_path(paper_payload, "swap_demo"))
                with fig2:
                    _render_svg_preview("High Load Kıyas Figürü", _resolve_figure_path(paper_payload, "high_load_compare"))
                with fig3:
                    _render_svg_preview("Dinamik Engel Figürü", _resolve_figure_path(paper_payload, "dynamic_obstacle"))

            with st.expander("Advanced", expanded=False):
                st.markdown("#### Dosya Çıktıları")
                st.code(json.dumps(paper_payload, indent=2, ensure_ascii=False), language="json")
                raw_rows = _read_csv_rows(paper_payload["raw_csv"])
                st.markdown("#### Ham suite tablosu")
                st.dataframe(raw_rows, width="stretch")


if __name__ == "__main__":
    main()
