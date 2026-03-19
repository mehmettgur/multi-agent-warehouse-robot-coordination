from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from warehouse_sim.loader import load_scenario
from warehouse_sim.models import PlannerConfig, RunResult, SimulationConfig, TickSnapshot
from warehouse_sim.paper_tables import PLANNER_LABELS
from warehouse_sim.scenario_catalog import scenario_path
from warehouse_sim.simulator import run_simulation

OUTPUT_FILES = {
    "swap_demo": "swap_demo.svg",
    "high_load_compare": "high_load_compare.svg",
    "dynamic_obstacle": "dynamic_obstacle.svg",
    "architecture_flow": "architecture_flow.svg",
    "experiment_pipeline": "experiment_pipeline.svg",
}


def _parse_heatmap(metrics: dict[str, object]) -> dict[tuple[int, int], int]:
    parsed: dict[tuple[int, int], int] = {}
    for key, value in metrics.get("congestion_heatmap", {}).items():
        try:
            x_str, y_str = str(key).split(",")
            parsed[(int(x_str), int(y_str))] = int(value)
        except ValueError:
            continue
    return parsed


def _select_snapshot(result: RunResult, prefer_blocked: bool = False) -> TickSnapshot:
    if not result.timeline:
        raise ValueError("Timeline boş, figür üretilemedi.")
    if prefer_blocked:
        for snapshot in result.timeline:
            if snapshot.blocked_cells:
                return snapshot
    return result.timeline[min(len(result.timeline) - 1, max(1, len(result.timeline) // 2))]


def _heat_fill(cell: tuple[int, int], heatmap: dict[tuple[int, int], int]) -> str:
    if cell not in heatmap or not heatmap:
        return "#f8fafc"
    max_count = max(heatmap.values())
    intensity = heatmap[cell] / max(1, max_count)
    red = 255
    green = int(246 - 96 * intensity)
    blue = int(238 - 150 * intensity)
    return f"rgb({red},{green},{blue})"


def _grid_svg(
    config: SimulationConfig,
    snapshot: TickSnapshot,
    title: str,
    subtitle: str,
    heatmap: dict[tuple[int, int], int] | None = None,
) -> str:
    cell_size = 38
    margin_x = 48
    margin_y = 70
    width = margin_x * 2 + config.width * cell_size
    height = margin_y + 60 + config.height * cell_size
    positions = snapshot.robot_positions
    blocked = set(snapshot.blocked_cells)
    pickups = set(config.stations.get("pickups", []))
    dropoffs = set(config.stations.get("dropoffs", []))
    robot_ids = sorted(positions)
    palette = ["#0ea5e9", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#eab308"]
    color_map = {rid: palette[idx % len(palette)] for idx, rid in enumerate(robot_ids)}

    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='{margin_x}' y='30' font-size='22' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>{escape(title)}</text>",
        f"<text x='{margin_x}' y='52' font-size='12' font-family='Arial, sans-serif' fill='#475569'>{escape(subtitle)}</text>",
    ]

    for y in range(config.height):
        for x in range(config.width):
            left = margin_x + x * cell_size
            top = margin_y + y * cell_size
            cell = (x, y)
            fill = "#f8fafc"
            stroke = "#cbd5e1"
            if cell in config.obstacles or cell in blocked:
                fill = "#334155"
                stroke = "#334155"
            elif heatmap is not None:
                fill = _heat_fill(cell, heatmap)
            if cell in pickups:
                fill = "#cffafe"
                stroke = "#0891b2"
            if cell in dropoffs:
                fill = "#ffedd5"
                stroke = "#ea580c"
            if cell in blocked:
                fill = "#111827"
                stroke = "#111827"

            pieces.append(
                f"<rect x='{left}' y='{top}' width='{cell_size - 2}' height='{cell_size - 2}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='1.2'/>"
            )

            label = ""
            label_fill = "#334155"
            if cell in config.obstacles or cell in blocked:
                label = "#"
                label_fill = "#f8fafc"
            elif cell in pickups:
                label = "P"
                label_fill = "#155e75"
            elif cell in dropoffs:
                label = "D"
                label_fill = "#9a3412"

            for rid, pos in positions.items():
                if pos == cell:
                    robot_fill = color_map[rid]
                    pieces.append(
                        f"<rect x='{left + 4}' y='{top + 4}' width='{cell_size - 10}' height='{cell_size - 10}' rx='8' fill='{robot_fill}'/>"
                    )
                    pieces.append(
                        f"<text x='{left + cell_size / 2 - 1}' y='{top + 22}' font-size='11' text-anchor='middle' font-family='Arial, sans-serif' font-weight='700' fill='#ffffff'>{escape(rid)}</text>"
                    )
                    label = ""
                    break

            if label:
                pieces.append(
                    f"<text x='{left + cell_size / 2 - 1}' y='{top + 23}' font-size='12' text-anchor='middle' font-family='Arial, sans-serif' font-weight='700' fill='{label_fill}'>{label}</text>"
                )

    legend_x = margin_x
    legend_y = height - 18
    pieces.append(
        f"<text x='{legend_x}' y='{legend_y}' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Tick {snapshot.tick} | Tamamlanan görev: {snapshot.completed_tasks} | Çakışma: {snapshot.collision_events}</text>"
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _metric_card(x: int, y: int, width: int, title: str, value: str, accent: str) -> str:
    return "".join(
        [
            f"<rect x='{x}' y='{y}' width='{width}' height='78' rx='16' fill='#f8fafc' stroke='#dbe4ef'/>",
            f"<text x='{x + 18}' y='{y + 26}' font-size='12' font-family='Arial, sans-serif' fill='#475569'>{escape(title)}</text>",
            f"<text x='{x + 18}' y='{y + 56}' font-size='26' font-family='Arial, sans-serif' font-weight='700' fill='{accent}'>{escape(value)}</text>",
        ]
    )


def _comparison_svg(baseline: RunResult, coordinated: RunResult, title: str) -> str:
    width = 940
    height = 420
    metrics_left = baseline.metrics.to_dict()
    metrics_right = coordinated.metrics.to_dict()
    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        f"<text x='48' y='34' font-size='24' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>{escape(title)}</text>",
        "<text x='48' y='58' font-size='12' font-family='Arial, sans-serif' fill='#475569'>A* + Greedy ile baseline ve koordineli mod karşılaştırması</text>",
        "<rect x='48' y='82' width='396' height='280' rx='18' fill='#fff7ed' stroke='#fed7aa'/>",
        "<rect x='496' y='82' width='396' height='280' rx='18' fill='#ecfeff' stroke='#a5f3fc'/>",
        "<text x='72' y='114' font-size='20' font-family='Arial, sans-serif' font-weight='700' fill='#9a3412'>Baseline</text>",
        "<text x='520' y='114' font-size='20' font-family='Arial, sans-serif' font-weight='700' fill='#155e75'>Koordineli</text>",
    ]

    left_cards = [
        ("Tamamlanan Görev", f"{metrics_left['completed_tasks']}/{metrics_left['total_tasks']}"),
        ("Çakışma", str(metrics_left['collision_count'])),
        ("Makespan", str(metrics_left['makespan'])),
        ("Throughput", f"{metrics_left['throughput']:.4f}"),
    ]
    right_cards = [
        ("Tamamlanan Görev", f"{metrics_right['completed_tasks']}/{metrics_right['total_tasks']}"),
        ("Çakışma", str(metrics_right['collision_count'])),
        ("Makespan", str(metrics_right['makespan'])),
        ("Throughput", f"{metrics_right['throughput']:.4f}"),
    ]

    for idx, (label, value) in enumerate(left_cards):
        col = idx % 2
        row = idx // 2
        pieces.append(_metric_card(72 + col * 188, 136 + row * 104, 164, label, value, "#9a3412"))
    for idx, (label, value) in enumerate(right_cards):
        col = idx % 2
        row = idx // 2
        pieces.append(_metric_card(520 + col * 188, 136 + row * 104, 164, label, value, "#155e75"))

    pieces.extend(
        [
            "<text x='48' y='392' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Koordineli modun ana kazanımı: çakışmaları sıfırlarken görev tamamlama oranını korumak veya artırmak.</text>",
            "</svg>",
        ]
    )
    return "".join(pieces)


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#64748b", width: int = 2) -> str:
    return "".join(
        [
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{width}' stroke-linecap='round'/>",
            f"<polygon points='{x2},{y2} {x2 - 10},{y2 - 5} {x2 - 10},{y2 + 5}' fill='{color}'/>",
        ]
    )


def _arch_box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body_lines: list[str],
    fill: str,
    stroke: str,
    title_fill: str = "#0f172a",
) -> str:
    pieces = [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='18' fill='{fill}' stroke='{stroke}' stroke-width='1.6'/>",
        f"<text x='{x + 16}' y='{y + 26}' font-size='16' font-family='Arial, sans-serif' font-weight='700' fill='{title_fill}'>{escape(title)}</text>",
    ]
    for idx, line in enumerate(body_lines):
        pieces.append(
            f"<text x='{x + 16}' y='{y + 50 + idx * 16}' font-size='11.5' font-family='Arial, sans-serif' fill='#334155'>{escape(line)}</text>"
        )
    return "".join(pieces)


def _step_box(x: int, y: int, w: int, h: int, step: str, text: str, fill: str, stroke: str) -> str:
    return "".join(
        [
            f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='16' fill='{fill}' stroke='{stroke}' stroke-width='1.4'/>",
            f"<circle cx='{x + 20}' cy='{y + 20}' r='12' fill='{stroke}'/>",
            f"<text x='{x + 20}' y='{y + 24}' text-anchor='middle' font-size='11' font-family='Arial, sans-serif' font-weight='700' fill='#ffffff'>{escape(step)}</text>",
            f"<text x='{x + 42}' y='{y + 24}' font-size='12.5' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>{escape(text)}</text>",
        ]
    )


def _architecture_flow_svg() -> str:
    width = 1240
    height = 860
    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        "<text x='48' y='34' font-size='26' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>End-to-End System Architecture and Tick-Level Control/Data Flow</text>",
        "<text x='48' y='58' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Architecture reconstructed directly from the implemented simulator, coordination logic, and experiment pipeline.</text>",
        "<text x='48' y='92' font-size='18' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>A. High-Level System Architecture</text>",
    ]

    pieces.append(
        _arch_box(
            48,
            120,
            212,
            128,
            "Scenario Config",
            [
                "JSON scenario files",
                "grid, robots, tasks, stations",
                "planner, allocator, events, seed",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )
    pieces.append(
        _arch_box(
            304,
            120,
            230,
            128,
            "Loader + Simulator",
            [
                "load_scenario(...)",
                "run_simulation(...)",
                "build GridMap / RobotState / TaskState",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )
    pieces.append(
        _arch_box(
            578,
            108,
            304,
            152,
            "CoordinatorAgent",
            [
                "tick loop orchestration",
                "policy dispatch (baseline / coordinated)",
                "conflict handling + task progress + stop condition",
            ],
            "#ecfeff",
            "#67e8f9",
            "#155e75",
        )
    )
    pieces.append(
        _arch_box(
            926,
            120,
            266,
            128,
            "Outputs",
            [
                "RunResult + TickSnapshot timeline",
                "MetricsReport",
                "paper CSV / LaTeX / SVG exports",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )

    pieces.append(
        _arch_box(
            524,
            300,
            198,
            110,
            "TaskAllocatorAgent",
            [
                "greedy or hungarian",
                "assign idle robots to released tasks",
            ],
            "#fefce8",
            "#fde68a",
            "#854d0e",
        )
    )
    pieces.append(
        _arch_box(
            752,
            300,
            220,
            110,
            "TrafficManagerAgent",
            [
                "prioritized planning",
                "reservation table",
                "dynamic priority / micro-replan",
            ],
            "#eff6ff",
            "#bfdbfe",
            "#1d4ed8",
        )
    )
    pieces.append(
        _arch_box(
            1002,
            300,
            190,
            110,
            "RobotAgent(s)",
            [
                "consume plan",
                "emit MOVE / WAIT",
            ],
            "#f5f3ff",
            "#ddd6fe",
            "#6d28d9",
        )
    )
    pieces.append(
        _arch_box(
            274,
            300,
            220,
            110,
            "EventEngine",
            [
                "temp_block",
                "stochastic_delay",
                "active blocked cells / forced waits",
            ],
            "#fff7ed",
            "#fed7aa",
            "#9a3412",
        )
    )
    pieces.append(
        _arch_box(
            48,
            300,
            196,
            110,
            "MetricsCollector",
            [
                "makespan, throughput, waits",
                "conflicts, replans, heatmap",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )

    pieces.extend(
        [
            _arrow(260, 184, 304, 184),
            _arrow(534, 184, 578, 184),
            _arrow(882, 184, 926, 184),
            _arrow(678, 260, 678, 300),
            _arrow(824, 260, 824, 300),
            _arrow(1097, 260, 1097, 300),
            _arrow(384, 260, 384, 300),
            _arrow(146, 300, 146, 248),
            _arrow(244, 355, 524, 355),
            _arrow(722, 355, 752, 355),
            _arrow(972, 355, 1002, 355),
            _arrow(1100, 410, 1100, 436),
            _arrow(1100, 436, 650, 436),
            _arrow(650, 436, 650, 260),
        ]
    )

    pieces.extend(
        [
            "<text x='48' y='496' font-size='18' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>B. Tick-Level Control and Data Flow</text>",
            "<rect x='48' y='516' width='1144' height='288' rx='22' fill='#f8fafc' stroke='#e2e8f0'/>",
        ]
    )

    step_specs = [
        (72, 548, 326, 58, "1", "Update events and task state", "#fff7ed", "#f97316"),
        (430, 548, 326, 58, "2", "Allocate released tasks", "#fefce8", "#eab308"),
        (788, 548, 372, 58, "3", "Plan intents (independent or reserved)", "#eff6ff", "#3b82f6"),
        (72, 632, 326, 58, "4", "Apply delays and blocked-cell overrides", "#fff7ed", "#f97316"),
        (430, 632, 326, 58, "5", "Detect conflicts and local micro-replan", "#f5f3ff", "#8b5cf6"),
        (788, 632, 372, 58, "6", "Fallback: lower-priority robots WAIT", "#ecfeff", "#06b6d4"),
        (256, 716, 326, 58, "7", "Execute MOVE / WAIT and update positions", "#ecfccb", "#65a30d"),
        (612, 716, 326, 58, "8", "Record snapshot, metrics, and stop if done", "#f8fafc", "#64748b"),
    ]
    for spec in step_specs:
        pieces.append(_step_box(*spec))

    pieces.extend(
        [
            _arrow(398, 577, 430, 577),
            _arrow(756, 577, 788, 577),
            _arrow(974, 606, 974, 632),
            _arrow(788, 661, 756, 661),
            _arrow(430, 661, 398, 661),
            _arrow(235, 690, 235, 716),
            _arrow(582, 745, 612, 745),
        ]
    )

    pieces.extend(
        [
            "<text x='72' y='790' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Core control loop in implementation: sense/update - assign - plan - resolve - act - record.</text>",
            "<text x='72' y='808' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Full coordinated mode uses reservation-based planning, optional dynamic priority, and localized replanning before deterministic WAIT fallback.</text>",
            "</svg>",
        ]
    )
    return "".join(pieces)


def _pipeline_box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body_lines: list[str],
    fill: str,
    stroke: str,
    title_fill: str = "#0f172a",
) -> str:
    return _arch_box(x, y, w, h, title, body_lines, fill, stroke, title_fill)


def _experiment_pipeline_svg() -> str:
    width = 1240
    height = 760
    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        "<text x='48' y='34' font-size='26' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>Reproducible Experiment Pipeline from Scenario JSON to Paper Artifacts</text>",
        "<text x='48' y='58' font-size='12' font-family='Arial, sans-serif' fill='#475569'>Implemented pipeline used to execute benchmark suites, aggregate outputs, and generate paper-ready tables and figures.</text>",
        "<text x='48' y='96' font-size='18' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>A. Inputs and Canonical Experiment Suites</text>",
    ]

    pieces.append(
        _pipeline_box(
            48,
            122,
            250,
            140,
            "Scenario JSON Inputs",
            [
                "core scenarios",
                "coordination-targeted scenarios",
                "dynamic and stochastic scenarios",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )
    pieces.append(
        _pipeline_box(
            340,
            122,
            270,
            140,
            "Runner Suite Selection",
            [
                "main",
                "allocator / planner / coordination",
                "robustness / all",
            ],
            "#eff6ff",
            "#bfdbfe",
            "#1d4ed8",
        )
    )
    pieces.append(
        _pipeline_box(
            652,
            122,
            250,
            140,
            "Simulation Execution",
            [
                "run_simulation(...)",
                "baseline / baseline_priority / coordinated",
                "fixed seed for deterministic replay",
            ],
            "#ecfeff",
            "#67e8f9",
            "#155e75",
        )
    )
    pieces.append(
        _pipeline_box(
            944,
            122,
            248,
            140,
            "Per-Run Outputs",
            [
                "RunResult",
                "timeline snapshots",
                "MetricsReport",
            ],
            "#f5f3ff",
            "#ddd6fe",
            "#6d28d9",
        )
    )

    pieces.extend(
        [
            _arrow(298, 192, 340, 192),
            _arrow(610, 192, 652, 192),
            _arrow(902, 192, 944, 192),
        ]
    )

    pieces.extend(
        [
            "<text x='48' y='330' font-size='18' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>B. Aggregation and Paper Artifact Generation</text>",
        ]
    )

    pieces.append(
        _pipeline_box(
            48,
            356,
            250,
            154,
            "Raw Exports",
            [
                "main_raw.csv / .json",
                "allocator_raw.csv / .json",
                "planner_raw.csv / .json",
                "coordination_raw.csv / .json",
                "robustness_raw.csv / .json",
            ],
            "#fff7ed",
            "#fed7aa",
            "#9a3412",
        )
    )
    pieces.append(
        _pipeline_box(
            340,
            356,
            270,
            154,
            "Paper Table Builder",
            [
                "generate_suite_tables(...)",
                "CSV summary tables",
                "LaTeX table exports",
            ],
            "#fefce8",
            "#fde68a",
            "#854d0e",
        )
    )
    pieces.append(
        _pipeline_box(
            652,
            356,
            250,
            154,
            "Figure Builder",
            [
                "generate_paper_figures(...)",
                "scenario snapshots",
                "comparison and architecture figures",
            ],
            "#ecfccb",
            "#bef264",
            "#3f6212",
        )
    )
    pieces.append(
        _pipeline_box(
            944,
            356,
            248,
            154,
            "Paper Artifacts",
            [
                "main_comparison.tex",
                "allocator / planner / coordination / robustness .tex",
                "SVG figures for paper and slides",
            ],
            "#f8fafc",
            "#cbd5e1",
        )
    )

    pieces.extend(
        [
            _arrow(173, 262, 173, 356),
            _arrow(298, 434, 340, 434),
            _arrow(610, 434, 652, 434),
            _arrow(902, 434, 944, 434),
        ]
    )

    pieces.extend(
        [
            "<text x='48' y='560' font-size='18' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>C. Reproducibility Guarantees Encoded in the Repository</text>",
            "<rect x='48' y='584' width='1144' height='118' rx='20' fill='#f8fafc' stroke='#e2e8f0'/>",
            "<text x='72' y='618' font-size='13' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>Deterministic configuration:</text>",
            "<text x='250' y='618' font-size='13' font-family='Arial, sans-serif' fill='#334155'>scenario seeds, fixed suite definitions, and canonical planner/allocation settings.</text>",
            "<text x='72' y='646' font-size='13' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>Standardized outputs:</text>",
            "<text x='250' y='646' font-size='13' font-family='Arial, sans-serif' fill='#334155'>raw CSV/JSON, summarized CSV, LaTeX tables, SVG figures, and figure manifest.</text>",
            "<text x='72' y='674' font-size='13' font-family='Arial, sans-serif' font-weight='700' fill='#0f172a'>Paper-ready workflow:</text>",
            "<text x='250' y='674' font-size='13' font-family='Arial, sans-serif' fill='#334155'>the same code path supports CLI experiments, UI paper export, and reproducible artifact regeneration.</text>",
            "</svg>",
        ]
    )
    return "".join(pieces)


def _write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _maybe_write_png(svg_content: str, target: Path) -> Path | None:
    try:
        import cairosvg  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    try:
        cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=str(target))
        return target
    except Exception:  # noqa: BLE001
        return None


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_paper_figures(output_dir: Path) -> dict[str, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = PlannerConfig(algorithm="astar", heuristic_weight=1.0)

    swap_config = load_scenario(scenario_path("narrow_corridor_swap"))
    swap_result = run_simulation(
        config=swap_config,
        mode="coordinated",
        seed=swap_config.seed,
        planner_override=planner,
        allocator_override="greedy",
    )
    swap_snapshot = _select_snapshot(swap_result)
    swap_svg = _grid_svg(
        config=swap_config,
        snapshot=swap_snapshot,
        title="Dar Koridor Demo",
        subtitle="Koordineli mod | A* | Greedy | çakışmasız geçiş örneği",
    )

    high_load_config = load_scenario(scenario_path("high_load_6r_30t"))
    high_load_baseline = run_simulation(
        config=high_load_config,
        mode="baseline",
        seed=high_load_config.seed,
        planner_override=planner,
        allocator_override="greedy",
    )
    high_load_coordinated = run_simulation(
        config=high_load_config,
        mode="coordinated",
        seed=high_load_config.seed,
        planner_override=planner,
        allocator_override="greedy",
    )
    comparison_svg = _comparison_svg(
        baseline=high_load_baseline,
        coordinated=high_load_coordinated,
        title="Yüksek Görev Yükünde Koordinasyon Etkisi",
    )

    dynamic_config = load_scenario(scenario_path("dynamic_obstacle"))
    dynamic_result = run_simulation(
        config=dynamic_config,
        mode="coordinated",
        seed=dynamic_config.seed,
        planner_override=planner,
        allocator_override="hungarian",
    )
    dynamic_snapshot = _select_snapshot(dynamic_result, prefer_blocked=True)
    dynamic_svg = _grid_svg(
        config=dynamic_config,
        snapshot=dynamic_snapshot,
        title="Dinamik Engel Senaryosu",
        subtitle="Koordineli mod | A* | Hungarian | geçici blokaj sırasında görünüm",
        heatmap=_parse_heatmap(dynamic_result.metrics.to_dict()),
    )

    outputs: dict[str, list[Path]] = {}
    figure_map = {
        "swap_demo": swap_svg,
        "high_load_compare": comparison_svg,
        "dynamic_obstacle": dynamic_svg,
        "architecture_flow": _architecture_flow_svg(),
        "experiment_pipeline": _experiment_pipeline_svg(),
    }
    for key, svg_content in figure_map.items():
        svg_path = output_dir / OUTPUT_FILES[key]
        _write_svg(svg_path, svg_content)
        outputs[key] = [svg_path]
        png_path = _maybe_write_png(svg_content, svg_path.with_suffix(".png"))
        if png_path is not None:
            outputs[key].append(png_path)

    manifest = {
        "planner": PLANNER_LABELS[planner.algorithm],
        "figures": {key: [str(path) for path in paths] for key, paths in outputs.items()},
    }
    manifest_path = output_dir / "figures_manifest.json"
    _write_manifest(manifest_path, manifest)
    outputs["manifest"] = [manifest_path]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Makale ve sunum için reproducible figür paketi üretir.")
    parser.add_argument("--output-dir", default="results/paper", help="Çıktı klasörü")
    args = parser.parse_args()

    outputs = generate_paper_figures(Path(args.output_dir))
    for key, paths in outputs.items():
        print(f"{key}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
