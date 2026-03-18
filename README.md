# Multi-Agent Warehouse Robot Coordination

2D grid tabanlı depo ortamında çoklu robot koordinasyonu simülasyonu.

Bu sürüm, proje anlatısını üç katmana ayırır:
- `Demo`: Baseline ve koordineli mod farkını hızlı göstermek için sade UI akışı
- `Paper Pack`: Makale için kanonik benchmark suite'leri, temiz CSV tabloları ve LaTeX çıktıları
- `Appendix / Advanced`: Dynamic event ve legacy senaryolar

## Plan Dosyaları
- `PLAN.md`: İlk MVP planı
- `PLAN2.MD`: Ablation ve genişletilmiş koordinasyon planı

## Ana Özellikler
- Ajan mimarisi:
  - `RobotAgent`
  - `TaskAllocatorAgent` (`greedy`, `hungarian`)
  - `TrafficManagerAgent` (prioritized planning + reservation table)
  - `CoordinatorAgent` (tick döngüsü, local micro-replan, metrik toplama)
- Planner seçenekleri:
  - `astar`
  - `dijkstra`
  - `weighted_astar`
- Coordinated mod:
  - vertex-time + edge-time rezervasyon
  - deterministic prioritized planning
  - local micro-replan fallback
- Event engine:
  - `temp_block`
  - `stochastic_delay`
- Paper/export hattı:
  - kanonik benchmark suite'leri
  - sade CSV tabloları
  - `.tex` tablo çıktıları
  - reproducible SVG figürler

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

Not: Editable kurulum yapmadıysan komutları `PYTHONPATH=src` ile çalıştır.

## Demo UI
```bash
streamlit run app/dashboard.py
```

UI iki sekmeden oluşur:
- `Demo`: varsayılan olarak `narrow_corridor_swap` ile baseline vs koordineli kıyas
- `Paper Pack`: kanonik benchmark suite'lerini çalıştırır, tabloları ve figürleri üretir

Demo tarafında varsayılan görünür senaryolar:
- `narrow_corridor_swap`
- `intersection_4way_crossing`
- `bottleneck_shelves`
- `high_load_6r_30t`

Appendix senaryoları:
- `dynamic_obstacle`
- `stochastic_delay`

Legacy senaryolar varsayılan olarak gizlidir:
- `narrow_corridor`
- `dense_tasks`

## CLI Kullanımı
### Tek koşum
```bash
python -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor_swap.json \
  --mode coordinated \
  --planner astar \
  --allocator greedy \
  --seed 17
```

### Baseline vs Coordinated kıyas
```bash
python -m warehouse_sim.runner \
  --scenario configs/scenarios/narrow_corridor_swap.json \
  --compare \
  --planner astar \
  --allocator greedy \
  --seed 17
```

### Genel ablation
```bash
python -m warehouse_sim.runner \
  --ablation \
  --scenarios configs/scenarios/narrow_corridor_swap.json configs/scenarios/intersection_4way_crossing.json \
  --output-dir results
```

## Paper Pack Komutları
### Ana karşılaştırma tablosu
```bash
python -m warehouse_sim.runner --suite main --latex
```

### Atayıcı ablation tablosu
```bash
python -m warehouse_sim.runner --suite allocator --latex
```

### Planlayıcı ablation tablosu
```bash
python -m warehouse_sim.runner --suite planner --latex
```

### Robustness appendix tablosu
```bash
python -m warehouse_sim.runner --suite robustness --seeds 11 17 23 31 37 --latex
```

### Tüm paper paketi
```bash
python -m warehouse_sim.runner --suite all --latex --figures
```

Varsayılan çıktı klasörü:
- `results/paper/main_comparison.csv`
- `results/paper/allocator_ablation.csv`
- `results/paper/planner_ablation.csv`
- `results/paper/robustness.csv`
- `results/paper/*.tex`
- `results/paper/*.svg`

Ham suite çıktıları:
- `results/paper/main_raw.csv`
- `results/paper/allocator_raw.csv`
- `results/paper/planner_raw.csv`
- `results/paper/robustness_raw.csv`
- `results/paper/all_raw.csv`

## Figür Üretimi
Sadece figür paketini üretmek istersen:
```bash
python -m warehouse_sim.figures --output-dir results/paper
```

Üretilen figürler:
- `results/paper/swap_demo.svg`
- `results/paper/high_load_compare.svg`
- `results/paper/dynamic_obstacle.svg`

Not: Ortamda `cairosvg` kuruluysa aynı figürlerin `.png` versiyonları da otomatik üretilir.

## Ham CSV'den Tablo / LaTeX Üretimi
```bash
python -m warehouse_sim.paper_tables \
  --input-csv results/paper/main_raw.csv \
  --output-dir results/paper/reexport \
  --latex
```

## Test
```bash
pytest
```

## Determinizm
- Aynı senaryo + aynı seed + aynı mode/planner/allocator kombinasyonu aynı sonucu üretir.
- Çoklu seed koşuları yalnız `stochastic_delay` robustness analizi için kullanılır.

## Notlar
- Ana anlatı `A* + coordinated planning` üzerinedir.
- `Weighted A*` ve `Dijkstra`, planner ablation içinde destekleyici analiz olarak tutulur.
- `Hungarian`, ana coordination tablosunda değil; ayrı allocator ablation tablosunda değerlendirilir.
- `narrow_corridor` senaryosu legacy kabul edilir ve kanonik paper/demo paketlerine dahil edilmez.
