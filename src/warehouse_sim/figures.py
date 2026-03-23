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


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#000000", width: float = 1.2) -> str:
    return "".join(
        [
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{width}' stroke-linecap='round'/>",
            f"<polygon points='{x2},{y2} {x2 - 7},{y2 - 4} {x2 - 7},{y2 + 4}' fill='{color}'/>",
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
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='3' fill='{fill}' stroke='{stroke}' stroke-width='1.0'/>",
        f"<text x='{x + 10}' y='{y + 19}' font-size='10.5' font-family='Helvetica, Arial, sans-serif' font-weight='700' fill='{title_fill}'>{escape(title)}</text>",
    ]
    for idx, line in enumerate(body_lines):
        pieces.append(
            f"<text x='{x + 10}' y='{y + 38 + idx * 12}' font-size='8.7' font-family='Helvetica, Arial, sans-serif' fill='#111827'>{escape(line)}</text>"
        )
    return "".join(pieces)


def _step_box(x: int, y: int, w: int, h: int, step: str, text: str, fill: str, stroke: str) -> str:
    return "".join(
        [
            f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='3' fill='{fill}' stroke='{stroke}' stroke-width='1.0'/>",
            f"<rect x='{x + 8}' y='{y + 8}' width='18' height='18' rx='2' fill='#111827'/>",
            f"<text x='{x + 17}' y='{y + 21}' text-anchor='middle' font-size='8.5' font-family='Helvetica, Arial, sans-serif' font-weight='700' fill='#ffffff'>{escape(step)}</text>",
            f"<text x='{x + 34}' y='{y + 21}' font-size='9.2' font-family='Helvetica, Arial, sans-serif' font-weight='700' fill='#111827'>{escape(text)}</text>",
        ]
    )


def _architecture_flow_svg() -> str:
    width = 1120
    height = 690
    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
    ]

    pieces.append(
        _arch_box(
            48,
            46,
            212,
            124,
            "Scenario Config",
            [
                "JSON scenarios",
                "grid / robots / tasks / events",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            304,
            46,
            230,
            124,
            "Loader + Simulator",
            [
                "load_scenario(...)",
                "run_simulation(...)",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            550,
            34,
            292,
            148,
            "CoordinatorAgent",
            [
                "tick orchestration",
                "policy dispatch",
                "conflict resolution",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            882,
            46,
            206,
            124,
            "Outputs",
            [
                "timeline + metrics",
                "CSV / LaTeX / SVG",
            ],
            "#ffffff",
            "#000000",
        )
    )

    pieces.append(
        _arch_box(
            486,
            242,
            186,
            104,
            "TaskAllocatorAgent",
            [
                "greedy / Hungarian",
                "task assignment",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            696,
            242,
            206,
            104,
            "TrafficManagerAgent",
            [
                "prioritized planning",
                "reservation table",
                "priority / micro-replan",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            920,
            242,
            116,
            104,
            "RobotAgent(s)",
            [
                "plan -> MOVE / WAIT",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            274,
            242,
            190,
            104,
            "EventEngine",
            [
                "temp_block",
                "stochastic_delay",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _arch_box(
            48,
            242,
            190,
            104,
            "MetricsCollector",
            [
                "makespan / waits",
                "conflicts / replans",
            ],
            "#ffffff",
            "#000000",
        )
    )

    pieces.extend(
        [
            _arrow(260, 108, 304, 108),
            _arrow(534, 108, 550, 108),
            _arrow(842, 108, 882, 108),
            _arrow(656, 182, 656, 242),
            _arrow(798, 182, 798, 242),
            _arrow(978, 170, 978, 242),
            _arrow(368, 170, 368, 242),
            _arrow(143, 242, 143, 170),
            _arrow(238, 294, 486, 294),
            _arrow(672, 294, 696, 294),
            _arrow(902, 294, 920, 294),
        ]
    )

    pieces.extend(
        [
            "<line x1='48' y1='384' x2='1072' y2='384' stroke='#000000' stroke-width='0.8' stroke-dasharray='5 5'/>",
        ]
    )

    step_specs = [
        (72, 420, 280, 48, "1", "update state", "#ffffff", "#000000"),
        (394, 420, 280, 48, "2", "allocate tasks", "#ffffff", "#000000"),
        (716, 420, 280, 48, "3", "plan intents", "#ffffff", "#000000"),
        (72, 500, 280, 48, "4", "apply events", "#ffffff", "#000000"),
        (394, 500, 280, 48, "5", "resolve conflicts", "#ffffff", "#000000"),
        (716, 500, 280, 48, "6", "WAIT fallback", "#ffffff", "#000000"),
        (232, 580, 280, 48, "7", "execute actions", "#ffffff", "#000000"),
        (554, 580, 280, 48, "8", "record snapshot", "#ffffff", "#000000"),
    ]
    for spec in step_specs:
        pieces.append(_step_box(*spec))

    pieces.extend(
        [
            _arrow(352, 444, 394, 444),
            _arrow(674, 444, 716, 444),
            _arrow(856, 468, 856, 500),
            _arrow(716, 524, 674, 524),
            _arrow(394, 524, 352, 524),
            _arrow(208, 548, 208, 580),
            _arrow(512, 604, 554, 604),
        ]
    )

    pieces.extend(
        [
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
    width = 1100
    height = 500
    pieces: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
    ]

    pieces.append(
        _pipeline_box(
            48,
            54,
            220,
            126,
            "Scenario JSON Inputs",
            [
                "scenario files",
                "fixed seeds",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            306,
            54,
            220,
            126,
            "Runner Suite Selection",
            [
                "main",
                "allocator / planner",
                "coordination / robustness / all",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            564,
            54,
            220,
            126,
            "Simulation Execution",
            [
                "run_simulation(...)",
                "baseline / priority / coordinated",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            822,
            54,
            220,
            126,
            "Per-Run Outputs",
            [
                "RunResult",
                "timeline + metrics",
            ],
            "#ffffff",
            "#000000",
        )
    )

    pieces.extend(
        [
            _arrow(268, 117, 306, 117),
            _arrow(526, 117, 564, 117),
            _arrow(784, 117, 822, 117),
        ]
    )

    pieces.extend(
        [
            "<line x1='48' y1='238' x2='1052' y2='238' stroke='#000000' stroke-width='0.8' stroke-dasharray='5 5'/>",
        ]
    )

    pieces.append(
        _pipeline_box(
            48,
            284,
            220,
            142,
            "Raw Exports",
            [
                "main_raw.csv / .json",
                "suite raw CSV / JSON",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            322,
            284,
            220,
            142,
            "Paper Table Builder",
            [
                "CSV summaries",
                "LaTeX tables",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            596,
            284,
            220,
            142,
            "Figure Builder",
            [
                "scenario snapshots",
                "comparison diagrams",
            ],
            "#ffffff",
            "#000000",
        )
    )
    pieces.append(
        _pipeline_box(
            870,
            284,
            230,
            142,
            "Paper Artifacts",
            [
                "main_comparison.tex",
                "planner / coordination / robustness .tex",
                "paper figures",
            ],
            "#ffffff",
            "#000000",
        )
    )

    pieces.extend(
        [
            _arrow(158, 180, 158, 284),
            _arrow(274, 355, 322, 355),
            _arrow(542, 355, 596, 355),
            _arrow(816, 355, 870, 355),
        ]
    )

    pieces.append("</svg>")
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
