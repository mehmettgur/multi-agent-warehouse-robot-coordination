from __future__ import annotations

import argparse
import csv
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
    "coordinated": "Koordineli",
}
ALLOCATOR_LABELS: dict[AllocatorName, str] = {
    "greedy": "Greedy",
    "hungarian": "Hungarian",
}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    return int(round(_to_float(value)))


def build_paper_tables(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    main_rows: list[dict[str, object]] = []
    appendix_rows: list[dict[str, object]] = []

    for row in rows:
        completed = _to_int(row.get("completed_tasks"))
        total = max(1, _to_int(row.get("total_tasks")))
        completion_ratio = completed / total

        planner = str(row.get("planner", "astar"))
        mode = str(row.get("mode", "coordinated"))
        allocator = str(row.get("allocator", "greedy"))

        main_rows.append(
            {
                "Senaryo": row.get("scenario", ""),
                "Mod": MODE_LABELS.get(mode, mode),
                "Planlayıcı": PLANNER_LABELS.get(planner, planner),
                "Atayıcı": ALLOCATOR_LABELS.get(allocator, allocator),
                "Tamamlama Oranı": f"{completed}/{total} (%{completion_ratio * 100:.1f})",
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


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_paper_tables(
    input_csv: Path,
    output_dir: Path,
    with_timestamp: bool = False,
) -> tuple[Path, Path]:
    rows = _read_csv(input_csv)
    main_rows, appendix_rows = build_paper_tables(rows)

    suffix = ""
    if with_timestamp:
        suffix = "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    main_path = output_dir / f"paper_main_table{suffix}.csv"
    appendix_path = output_dir / f"paper_appendix_table{suffix}.csv"
    _write_csv(main_path, main_rows)
    _write_csv(appendix_path, appendix_rows)
    return main_path, appendix_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation CSV çıktısından makale için sade tablolar üretir.",
    )
    parser.add_argument("--input-csv", required=True, help="Ham ablation CSV yolu")
    parser.add_argument("--output-dir", default="results", help="Çıktı klasörü")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Dosya adlarına UTC zaman eki ekle",
    )
    args = parser.parse_args()

    main_path, appendix_path = generate_paper_tables(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        with_timestamp=args.timestamp,
    )
    print(f"Main table: {main_path}")
    print(f"Appendix table: {appendix_path}")


if __name__ == "__main__":
    main()
