# PLAN2 — Algoritma Ablation + Gelismis Koordinasyon + Benchmark/UI Genisletme

## Ozet
Bu plan, mevcut projeyi “deneysel calisma + guclu demo” seviyesine tasir. Secilen kapsam kararlari: `Dijkstra + Weighted A*` ablation, `Hungarian min-cost assignment`, `dynamic obstacle + stochastic delay` benchmark setine dahil. Hedef cikti: UI’dan algoritma/koordinasyon/senaryo secilebilir, baseline vs coordinated kiyaslari ve ablation tablolari uretilebilir, metrikler akademik olarak genisletilmis olur.

## Kapsam Kararlari (Kilitleme)
1. Arama algoritmalari: `astar`, `dijkstra`, `weighted_astar`.
2. Allocation yukseltmesi: `hungarian` (greedy ile karsilastirmali).
3. Belirsizlik: hem `temp_block` hem `stochastic_delay` aktif olarak uygulanacak.
4. Ablation sunumu: hem UI tablosu hem CSV/JSON ciktisi uretilecek.
5. Geri uyumluluk: mevcut senaryo JSON’lari alan eklenmeden de calismaya devam edecek.

## Mevcut Durumdan Tureyen Entegrasyon Noktalari
1. Yol planlama su an `src/warehouse_sim/pathfinding.py` icindeki tek A* fonksiyonuna bagli.
2. Task allocation su an `src/warehouse_sim/agents/task_allocator_agent.py` icinde greedy ETA.
3. Koordinasyon su an `src/warehouse_sim/agents/traffic_manager_agent.py` ve `src/warehouse_sim/agents/coordinator_agent.py` uzerinde.
4. Metrikler su an `src/warehouse_sim/metrics.py` icinde temel set.
5. UI su an `app/dashboard.py` icinde simulasyon ve replay odakli.

## Fazli Implementasyon Plani

## Faz 0 — Plan Ayristirma ve Dosya Organizasyonu
1. `PLAN2.MD` dosyasi repo kokune eklenecek ve bu plan icerigi birebir yazilacak.
2. Mevcut `PLAN.md` korunacak; README’ye “Plan dosyalari” kisa notu eklenecek.
3. Her faz sonunda kullanici istegi geregi ornek commit mesaji uretimi standartlastirilacak.

## Faz 1 — Planner Abstraction ve Algoritma Ablation Cekirdegi
1. `src/warehouse_sim/models.py` icine yeni tipler eklenecek: `PlannerAlgorithm`, `PlannerConfig`.
2. `src/warehouse_sim/pathfinding.py` genellestirilecek: ortak space-time search motoru + wrapper fonksiyonlar.
3. `astar`: `f = g + h`; `dijkstra`: `h = 0`; `weighted_astar`: `f = g + w*h`.
4. `weighted_astar` icin varsayilan `w = 1.4`, UI/CLI araligi `1.0–3.0`.
5. `src/warehouse_sim/policies/baseline_policy.py` ve `src/warehouse_sim/agents/traffic_manager_agent.py` planner config ile calisacak hale getirilecek.
6. Planner diagnostics eklenecek: `expanded_nodes`, `planning_time_ms`, `path_cost`.
7. Bu diagnostics run bazinda toplanip rapora aktarilacak.

## Faz 2 — Koordinasyon Guclendirme
1. Dinamik onceliklendirme skoru tek formulde kilitlenecek: `priority = active_task_flag, -blocking_score, -wait_streak, rotation_rank, eta`.
2. `src/warehouse_sim/agents/coordinator_agent.py` icine local conflict resolver eklenecek.
3. Local micro-replan kurali: yalniz cakisan robotlar icin `horizon=6` kisa yeniden planlama; diger robotlar sabit rezervasyon gibi kabul edilecek.
4. Cozulmeyen durumda deterministic fallback: en yuksek oncelik hareket eder, digerleri WAIT.
5. Coordinated modda fiziksel carpisma uygulanmasina izin verilmeyecek; `collision_count` 0 hedefi korunacak.
6. `replanning_count` artik genel replan + micro-replan toplamini ayri alanlarla raporlayacak.

## Faz 3 — Task Allocation Upgrade (Hungarian)
1. `src/warehouse_sim/agents/task_allocator_agent.py` policy destekleyecek: `greedy` ve `hungarian`.
2. Hungarian icin saf Python implementasyon eklenecek: `src/warehouse_sim/optimization/hungarian.py`.
3. Maliyet matrisi tanimi kilitlenecek: `cost = Manhattan(robot_pos, task_pickup)`.
4. Dikdortgen matris durumu desteklenecek: dummy gorev/robot padding ile cozum.
5. Tie-break deterministik olacak: `robot_id`, sonra `task_id`.
6. Varsayilan coordinated allocation policy: `hungarian`; baseline policy: `greedy`.

## Faz 4 — Olay Motoru ve Belirsizlik
1. `src/warehouse_sim/events.py` eklenecek ve Coordinator tick dongusune baglanacak.
2. Event schema kilidi:
   - `temp_block`: `{"type":"temp_block","cell":[x,y],"start_tick":k,"end_tick":m}`
   - `stochastic_delay`: `{"type":"stochastic_delay","probability":p,"start_tick":k,"end_tick":m,"robot_ids":[...]}`
3. Active dynamic block hucreleri planlama validasyonuna dahil edilecek.
4. Delay tetiklenirse robot o tick zorunlu WAIT yapacak.
5. RNG determinizmligi korunacak: tick basina robotlar `sorted(robot_id)` sirasiyla orneklenecek.
6. Yeni metrikler: `delay_event_count`, `dynamic_block_replans`.

## Faz 5 — Benchmark Senaryo Paketi (6 Senaryo)
1. `configs/scenarios/` altinda benchmark seti olusturulacak:
   - `narrow_corridor_swap.json`
   - `intersection_4way_crossing.json`
   - `bottleneck_shelves.json`
   - `high_load_6r_30t.json`
   - `dynamic_obstacle.json`
   - `stochastic_delay.json`
2. Mevcut `narrow_corridor.json` ve `dense_tasks.json` korunacak; benchmark seti ayrica listelenecek.
3. Her senaryoda seed ve max_tick sabitlenecek; kiyaslar tekrar uretilebilir olacak.
4. `dynamic_obstacle` ve `stochastic_delay` senaryolarinda event payload zorunlu olacak.

## Faz 6 — Metriklerin Akademik Genisletilmesi
1. `src/warehouse_sim/models.py` icindeki `MetricsReport` genisletilecek:
   - `throughput`
   - `fairness_task_std`
   - `fairness_task_cv`
   - `tasks_completed_per_robot`
   - `congestion_heatmap`
   - `planner_expanded_nodes_total`
   - `planner_time_ms_total`
2. `throughput = completed_tasks / max(1, makespan)` olarak hesaplanacak.
3. Fairness robot basina tamamlanan gorev dagilimi uzerinden hesaplanacak.
4. Congestion heatmap hucre ziyaret frekansi olarak sparse formatta saklanacak.
5. Runner JSON/CSV ciktilari yeni alanlari tasiyacak.

## Faz 7 — UI Ablation ve Gelismis Gorsellestirme
1. `app/dashboard.py` icine yeni “Ablation” bolumu eklenecek.
2. UI kontrolleri:
   - Planner secimi (`astar`, `dijkstra`, `weighted_astar`)
   - Weighted A* slider (`w`)
   - Allocation policy secimi (`greedy`, `hungarian`)
   - Scenario dropdown
   - Baseline/Coordinated toggle
   - Playback speed slider
   - Heatmap overlay toggle
3. Ablation tablosu ayni ekranda uretilecek: scenario × mode × planner × allocator.
4. Sonuclar UI’da tablo/bar chart olarak ve `results/ablation_*.csv` olarak disa verilecek.
5. JSON yogunlugu varsayilan gorunumden kaldirilacak; sadece advanced expander’da kalacak.

## Faz 8 — CLI/Runner Deney Akisi
1. `src/warehouse_sim/runner.py` yeni argumanlar alacak:
   - `--planner`
   - `--heuristic-weight`
   - `--allocator`
   - `--ablation`
   - `--scenarios`
2. `--ablation` modu, secili senaryolar uzerinde tum kombinasyonlari kosup tek CSV/JSON rapor uretecek.
3. Standart `--compare` akisi bozulmadan korunacak.

## Public API / Interface / Type Degisiklikleri
1. `run_simulation(config, mode, seed=None, planner_override=None, allocator_override=None) -> RunResult`.
2. `SimulationConfig` yeni alanlar:
   - `planner: PlannerConfig`
   - `allocator_policy: AllocationPolicy`
   - `events: list[dict]` mevcut alan aktif kullanilacak.
3. `TaskAllocatorAgent.assign_tasks(...)` yeni parametre: `policy`.
4. `TrafficManagerAgent.plan_and_reserve(...)` planner config alacak.
5. `MetricsCollector.finalize(...)` yeni akademik metrikleri dolduracak.

## Test Stratejisi ve Senaryolar
1. Unit: Dijkstra sonuc maliyeti = A* (w=1, ayni heuristic kosulunda).
2. Unit: Weighted A* path cost, A*’dan dusuk olmayacak; expanded node sayisi cogu vakada daha az olacak.
3. Unit: Hungarian bilinen maliyet matrisinde global optimum atamayi dondurecek.
4. Unit: Dynamic obstacle aktifken blocked cell’e plan uretilmeyecek.
5. Unit: Stochastic delay ayni seed ile birebir ayni delay olaylarini uretecek.
6. Integration: 6 benchmark senaryosunda `baseline` ve `coordinated` kosulari tamamlanacak.
7. Integration: Coordinated modda dar koridor ve intersection senaryolarinda `collision_count == 0`.
8. Integration: Ablation runner secili kombinasyonlar icin CSV/JSON uretecek.
9. UI smoke: planner/allocator secimleri simulasyon sonuclarina yansiyacak; heatmap overlay cizilecek.

## Kabul Kriterleri
1. UI’dan planner secimi ve weighted parametre degisimi gercek simulasyonu etkiliyor olmali.
2. Ablation tablosu en az `astar`, `dijkstra`, `weighted_astar` satirlarini uretiyor olmali.
3. Hungarian policy, greedy’e gore en az bazi senaryolarda toplam ETA maliyetini dusuruyor olmali.
4. Dynamic obstacle ve stochastic delay senaryolari deterministik seed ile tekrar uretilebilir olmali.
5. Yeni metrikler (`throughput`, `fairness`, `congestion`) JSON/CSV/UI’da gorunur olmali.
6. Mevcut testler kirilmadan yeni testler gecmeli.

## Varsayimlar ve Varsayilanlar
1. Python surumu `>=3.10` korunacak.
2. Agir dis bagimlilik eklenmeyecek; Hungarian saf Python ile yazilacak.
3. Weighted A* varsayilani `w=1.4`; `w=1.0` A* davranisina esit kabul edilecek.
4. Baseline policy sade tutulacak; Hungarian varsayilan olarak coordinated tarafta kullanilacak.
5. Heatmap sayaci MOVE ve WAIT her ikisinde de hucre isgalini sayacak.
6. Plan Mode kurali nedeniyle dosya yazimi bu asamada yapilmaz; `PLAN2.MD` olusturma execute asamasinin ilk adimidir.

## EXECUTE Asamasi Checklist
- [ ] `PLAN2.MD` dosyasini bu planla repo kokune yaz.
- [ ] Planner abstraction + `dijkstra` + `weighted_astar` implement et.
- [ ] Baseline ve coordinated planner-config uyumunu tamamla.
- [ ] Hungarian allocator policy’yi entegre et.
- [ ] Local micro-replan conflict resolver ekle.
- [ ] Event engine (`temp_block`, `stochastic_delay`) entegre et.
- [ ] 6 benchmark senaryoyu `configs/scenarios/` altina ekle.
- [ ] Akademik metrikleri `MetricsReport` ve collector’a ekle.
- [ ] Runner/CLI ablation modunu ekle.
- [ ] Dashboard’a planner/allocator/ablation/heatmap/speed kontrollerini ekle.
- [ ] Unit + integration + UI smoke testlerini tamamla.
- [ ] README’yi yeni deney akislari ve metrik aciklamalariyla guncelle.
- [ ] Her faz sonrasi ornek commit mesaji uret.
