# Multi-Agent Warehouse Robot Coordination

2D grid depo ortaminda coklu robot koordinasyonu simulasyonu.

## Plan Dosyalari
- `PLAN.md`: Ilk MVP plani.
- `PLAN2.MD`: Ablation + gelismis koordinasyon + benchmark/UI genisletme plani.

## Ozellikler
- Ajan mimarisi:
  - `RobotAgent`
  - `TaskAllocatorAgent` (`greedy`, `hungarian`)
  - `TrafficManagerAgent` (prioritized planning + reservation table)
  - `CoordinatorAgent` (tick dongusu, local micro-replan, metrik toplama)
- Planner secenekleri:
  - `astar`
  - `dijkstra`
  - `weighted_astar` (`w` ayarlanabilir)
- Coordinated mod:
  - vertex-time + edge-time rezervasyon
  - local conflict resolver + deterministic fallback
- Olay motoru:
  - `temp_block`
  - `stochastic_delay`
- Benchmark senaryolari (6 adet):
  - `narrow_corridor_swap`
  - `intersection_4way_crossing`
  - `bottleneck_shelves`
  - `high_load_6r_30t`
  - `dynamic_obstacle`
  - `stochastic_delay`
- Akademik metrikler:
  - makespan, throughput, fairness, congestion heatmap
  - planner diagnostics (expanded nodes, planning time, path cost)

## Gereksinimler
- Python `>=3.10`

## Kurulum
### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,ui]
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,ui]
```

Not: Paket editable kurulmazsa komutlara `PYTHONPATH=src` prefix ekleyin.

## CLI Kullanim
### Tek kosum
```bash
python -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor_swap.json \
  --mode coordinated \
  --planner astar \
  --allocator hungarian \
  --seed 17
```

### Baseline vs Coordinated kiyas
```bash
python -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor_swap.json \
  --compare \
  --planner weighted_astar \
  --heuristic-weight 1.4 \
  --allocator hungarian \
  --seed 17
```

### Ablation matrix
```bash
python -m warehouse_sim.runner \
  --ablation \
  --scenarios configs/scenarios/narrow_corridor_swap.json configs/scenarios/intersection_4way_crossing.json \
  --output-dir results
```

Uretilen ciktilar:
- `results/*_comparison.json`, `results/*_comparison.csv`
- `results/ablation_*.json`, `results/ablation_*.csv`

### Makale tablosu uretimi (sade CSV)
```bash
python -m warehouse_sim.paper_tables \
  --input-csv results/ablation_YYYYMMDD_HHMMSS.csv \
  --output-dir results
```

Uretilen dosyalar:
- `results/paper_main_table.csv`
- `results/paper_appendix_table.csv`

## Web UI
```bash
streamlit run app/dashboard.py
```

UI icerigi:
- Scenario secimi
- Single run / baseline vs coordinated secimi
- Planner secimi + weighted A* slider
- Allocator secimi
- Replay hiz kontrolu
- Heatmap overlay toggle
- Ablation tablosu ve kayit
- Opsiyonel JSON scenario editor (expander)

## Test
```bash
pytest
```

## Determinizm
- Ayni senaryo + ayni seed + ayni mode/planner/allocator kombinasyonu ayni sonucu uretir.

## Not
- Coordinated modda fiziksel carpisma metrigi `collision_count` hedefi 0'dir.
