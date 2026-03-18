from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

PlannerName = str
ModeName = str
AllocatorName = str

PLANNER_LABELS: dict[PlannerName, str] = {
    "astar": "A*",
    "dijkstra": "Dijkstra",
    "weighted_astar": "A* (Ağırlıklı)",
}
MODE_LABELS: dict[ModeName, str] = {
    "baseline": "Baseline",
    "baseline_priority": "Öncelikli Baseline",
    "coordinated": "Koordineli",
}
ALLOCATOR_LABELS: dict[AllocatorName, str] = {
    "greedy": "Greedy",
    "hungarian": "Hungarian",
}

SUITE_OUTPUTS: dict[str, str] = {
    "main": "main_comparison",
    "allocator": "allocator_ablation",
    "planner": "planner_ablation",
    "coordination": "coordination_ablation",
    "robustness": "robustness",
}
SUITE_CAPTIONS: dict[str, str] = {
    "main": "Temel senaryolarda baseline ve koordineli mod karşılaştırması.",
    "allocator": "Koordineli mod altında görev atama politikalarının karşılaştırması.",
    "planner": "Koordineli mod altında planlayıcı algoritmalarının karşılaştırması.",
    "coordination": "Koordinasyon bileşenlerinin ve daha adil baseline varyantının karşılaştırması.",
    "robustness": "Dinamik olaylar altında koordineli mod dayanıklılık özeti.",
}
COORDINATION_LABELS: dict[str, str] = {
    "independent": "Bağımsız Baseline",
    "priority_static": "Öncelikli Baseline (Statik Öncelik)",
    "vertex_only": "Koordineli (Edge Kapalı)",
    "static_priority": "Koordineli (Statik Öncelik)",
    "no_micro_replan": "Koordineli (Micro-Replan Kapalı)",
    "full": "Koordineli (Tam)",
}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    return int(round(_to_float(value)))


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_completion_ratio(completed: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    percent = (completed / total) * 100.0
    return f"{completed}/{total} (%{percent:.1f})"


def _format_float(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _format_mean_std(values: list[float], digits: int = 2) -> str:
    if not values:
        return "0"
    if len(values) == 1:
        return _format_float(values[0], digits)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return f"{_format_float(mean, digits)} ± {_format_float(std, digits)}"


def build_generic_tables(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    main_rows: list[dict[str, object]] = []
    appendix_rows: list[dict[str, object]] = []

    for row in rows:
        completed = _to_int(row.get("completed_tasks"))
        total = max(1, _to_int(row.get("total_tasks")))

        planner = str(row.get("planner", "astar"))
        mode = str(row.get("mode", "coordinated"))
        allocator = str(row.get("allocator", "greedy"))

        main_rows.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Mod": MODE_LABELS.get(mode, mode),
                "Planlayıcı": PLANNER_LABELS.get(planner, planner),
                "Atayıcı": ALLOCATOR_LABELS.get(allocator, allocator),
                "Tamamlama Oranı": _format_completion_ratio(completed, total),
                "Çakışma": _to_int(row.get("collision_count")),
                "Makespan": _to_int(row.get("makespan")),
                "Throughput": round(_to_float(row.get("throughput")), 4),
                "Ortalama Görev Süresi": round(_to_float(row.get("avg_task_completion_time")), 3),
                "Bekleme": _to_int(row.get("wait_count")),
            }
        )

        appendix_rows.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Mod": MODE_LABELS.get(mode, mode),
                "Planlayıcı": PLANNER_LABELS.get(planner, planner),
                "Atayıcı": ALLOCATOR_LABELS.get(allocator, allocator),
                "Genişletilen Düğüm": _to_int(row.get("planner_expanded_nodes_total")),
                "Planlama Süresi (ms)": round(_to_float(row.get("planner_time_ms_total")), 3),
                "Plan Yol Maliyeti": _to_int(row.get("planner_path_cost_total")),
                "Adillik Std": round(_to_float(row.get("fairness_task_std")), 4),
                "Adillik CV": round(_to_float(row.get("fairness_task_cv")), 4),
                "Gecikme Olayı": _to_int(row.get("delay_event_count")),
                "Dinamik Blok Replan": _to_int(row.get("dynamic_block_replans")),
            }
        )

    return main_rows, appendix_rows


def build_main_comparison_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for row in rows:
        completed = _to_int(row.get("completed_tasks"))
        total = max(1, _to_int(row.get("total_tasks")))
        mode = str(row.get("mode", "coordinated"))
        table.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Mod": MODE_LABELS.get(mode, mode),
                "Tamamlama Oranı": _format_completion_ratio(completed, total),
                "Çakışma": _to_int(row.get("collision_count")),
                "Makespan": _to_int(row.get("makespan")),
                "Throughput": round(_to_float(row.get("throughput")), 4),
                "Bekleme": _to_int(row.get("wait_count")),
                "Ortalama Görev Süresi": round(_to_float(row.get("avg_task_completion_time")), 3),
            }
        )
    return table


def build_allocator_ablation_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for row in rows:
        allocator = str(row.get("allocator", "greedy"))
        table.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Atayıcı": ALLOCATOR_LABELS.get(allocator, allocator),
                "Makespan": _to_int(row.get("makespan")),
                "Throughput": round(_to_float(row.get("throughput")), 4),
                "Toplam Yol": _to_int(row.get("total_path_length")),
                "Bekleme": _to_int(row.get("wait_count")),
            }
        )
    return table


def build_planner_ablation_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for row in rows:
        planner = str(row.get("planner", "astar"))
        table.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Planlayıcı": PLANNER_LABELS.get(planner, planner),
                "Makespan": _to_int(row.get("makespan")),
                "Throughput": round(_to_float(row.get("throughput")), 4),
                "Genişletilen Düğüm": _to_int(row.get("planner_expanded_nodes_total")),
                "Planlama Süresi (ms)": round(_to_float(row.get("planner_time_ms_total")), 3),
            }
        )
    return table


def build_coordination_ablation_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for row in rows:
        completed = _to_int(row.get("completed_tasks"))
        total = max(1, _to_int(row.get("total_tasks")))
        variant = str(row.get("coordination_variant", "full"))
        table.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Varyant": COORDINATION_LABELS.get(variant, variant),
                "Mod": MODE_LABELS.get(str(row.get("mode", "coordinated")), str(row.get("mode", "coordinated"))),
                "Tamamlama Oranı": _format_completion_ratio(completed, total),
                "Çakışma": _to_int(row.get("collision_count")),
                "Makespan": _to_int(row.get("makespan")),
                "Throughput": round(_to_float(row.get("throughput")), 4),
                "Bekleme": _to_int(row.get("wait_count")),
            }
        )
    return table


def build_robustness_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        scenario = str(row.get("scenario", ""))
        grouped.setdefault(scenario, []).append(row)

    table: list[dict[str, object]] = []
    for scenario in sorted(grouped):
        scenario_rows = sorted(grouped[scenario], key=lambda item: _to_int(item.get("seed")))
        completed_values = [_to_int(item.get("completed_tasks")) for item in scenario_rows]
        total_values = [_to_int(item.get("total_tasks")) for item in scenario_rows]
        collisions = [_to_float(item.get("collision_count")) for item in scenario_rows]
        makespans = [_to_float(item.get("makespan")) for item in scenario_rows]
        throughputs = [_to_float(item.get("throughput")) for item in scenario_rows]
        waits = [_to_float(item.get("wait_count")) for item in scenario_rows]
        seed_list = ", ".join(str(_to_int(item.get("seed"))) for item in scenario_rows)

        if len(scenario_rows) == 1:
            completed = completed_values[0]
            total = max(1, total_values[0])
            completion_text = _format_completion_ratio(completed, total)
        else:
            completed_mean = sum(completed_values) / len(completed_values)
            total = max(1, total_values[0])
            completion_text = f"{_format_float(completed_mean, 2)}/{total}"

        table.append(
            {
                "Senaryo": scenario,
                "Seedler": seed_list,
                "Tamamlama": completion_text,
                "Çakışma": _format_mean_std(collisions, 2),
                "Makespan": _format_mean_std(makespans, 2),
                "Throughput": _format_mean_std(throughputs, 4),
                "Bekleme": _format_mean_std(waits, 2),
            }
        )
    return table


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_label(value: str) -> str:
    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", ":"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def render_latex_table(
    rows: list[dict[str, object]],
    caption: str,
    label: str,
) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    alignment = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(_latex_escape(row.get(header, "")) for header in headers) + r" \\",
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{tab:{_latex_label(label)}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_latex_table(
    path: Path,
    rows: list[dict[str, object]],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_latex_table(rows, caption=caption, label=label), encoding="utf-8")


def infer_suites(rows: list[dict[str, object]]) -> list[str]:
    suites = [str(row.get("suite", "")).strip() for row in rows]
    return [suite for suite in dict.fromkeys(suites) if suite]


def _suite_table_rows(suite: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if suite == "main":
        return build_main_comparison_table(rows)
    if suite == "allocator":
        return build_allocator_ablation_table(rows)
    if suite == "planner":
        return build_planner_ablation_table(rows)
    if suite == "coordination":
        return build_coordination_ablation_table(rows)
    if suite == "robustness":
        return build_robustness_table(rows)
    raise ValueError(f"Unsupported suite: {suite}")


def generate_suite_tables(
    rows: list[dict[str, object]],
    output_dir: Path,
    with_latex: bool = False,
    with_timestamp: bool = False,
) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    suffix = ""
    if with_timestamp:
        suffix = "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    suites = infer_suites(rows)
    for suite in suites:
        suite_rows = [row for row in rows if str(row.get("suite", "")) == suite]
        table_rows = _suite_table_rows(suite, suite_rows)
        stem = SUITE_OUTPUTS[suite]
        csv_path = output_dir / f"{stem}{suffix}.csv"
        write_csv(csv_path, table_rows)
        outputs.setdefault(suite, []).append(csv_path)

        if with_latex:
            tex_path = output_dir / f"{stem}{suffix}.tex"
            write_latex_table(
                tex_path,
                table_rows,
                caption=SUITE_CAPTIONS[suite],
                label=stem,
            )
            outputs[suite].append(tex_path)
    return outputs


def generate_paper_tables(
    input_csv: Path,
    output_dir: Path,
    with_timestamp: bool = False,
    with_latex: bool = False,
) -> tuple[Path, Path] | dict[str, list[Path]]:
    rows = _read_csv(input_csv)
    suites = infer_suites(rows)
    if suites:
        return generate_suite_tables(
            rows=rows,
            output_dir=output_dir,
            with_latex=with_latex,
            with_timestamp=with_timestamp,
        )

    main_rows, appendix_rows = build_generic_tables(rows)

    suffix = ""
    if with_timestamp:
        suffix = "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    main_path = output_dir / f"paper_main_table{suffix}.csv"
    appendix_path = output_dir / f"paper_appendix_table{suffix}.csv"
    write_csv(main_path, main_rows)
    write_csv(appendix_path, appendix_rows)

    if with_latex:
        write_latex_table(
            output_dir / f"paper_main_table{suffix}.tex",
            main_rows,
            caption="Makale için ana sonuç tablosu.",
            label="paper_main_table",
        )
        write_latex_table(
            output_dir / f"paper_appendix_table{suffix}.tex",
            appendix_rows,
            caption="Makale için teknik ek tablo.",
            label="paper_appendix_table",
        )

    return main_path, appendix_path


def _print_outputs(outputs: dict[str, list[Path]] | tuple[Path, Path]) -> None:
    if isinstance(outputs, tuple):
        print(f"Main table: {outputs[0]}")
        print(f"Appendix table: {outputs[1]}")
        return

    for suite, paths in outputs.items():
        joined = ", ".join(str(path) for path in paths)
        print(f"{suite}: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV çıktılarından makale için sade tablolar ve isteğe bağlı LaTeX üretir.",
    )
    parser.add_argument("--input-csv", required=True, help="Ham CSV yolu")
    parser.add_argument("--output-dir", default="results", help="Çıktı klasörü")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Dosya adlarına UTC zaman eki ekle",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="CSV ile birlikte .tex tablo dosyaları da üret",
    )
    args = parser.parse_args()

    outputs = generate_paper_tables(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        with_timestamp=args.timestamp,
        with_latex=args.latex,
    )
    _print_outputs(outputs)


if __name__ == "__main__":
    main()
