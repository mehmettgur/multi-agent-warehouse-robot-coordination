# Multi-Agent Depo Robot Koordinasyonu — Uçtan Uca Implementasyon Planı

## Özet
Bu plan, projeyi sıfırdan `Python 3.11+` ile geliştirip ders kapsamındaki tüm zorunlu kavramları gösterilebilir hale getirir: akıllı ajan mimarisi, A* (Manhattan), heuristic tabanlı task allocation, CSP-benzeri çakışma kısıtları (vertex+edge), baseline vs coordinated kıyası, deterministik tekrar üretilebilir deneyler ve minimal Web UI demo.  
Seçilen hedef kapsam: `MVP deterministic`, `baseline=greedy no-reservation`, `UI=minimal dashboard`, `belirsizlik/replanning=post-MVP extension`.

## Ders Kapsamı Eşlemesi
1. Akıllı ajanlar: `RobotAgent`, `TaskAllocatorAgent`, `TrafficManagerAgent`, `CoordinatorAgent`.
2. Arama yöntemleri: zaman-genişletilmiş A* (`state=(x,y,t)`) + Manhattan heuristic + `WAIT`.
3. Akıllı arama/heuristik: task allocation’da ETA tabanlı greedy seçim.
4. Kısıt optimizasyonu/CSP: obstacle, vertex collision, edge-swap collision, task exclusivity, pickup-before-dropoff.
5. Belirsizlik/olasılık: post-MVP olarak geçici blokaj/gecikme olayı + sınırlı replanning.

## 1) Fazlara Bölünmüş MVP Planı

### Faz 1 — Temel Simülasyon ve A* Çekirdeği
1. Proje iskeleti ve paket yapısını kur.
2. Grid, obstacle, station, robot, task veri modellerini tanımla.
3. Deterministik RNG/seed altyapısı ekle.
4. A* (Manhattan) + `WAIT` aksiyonu ile tek robot rota planlamasını bitir.
5. Tick tabanlı temel simülasyon döngüsünü (sense→plan→act) çıkar.
6. Tek senaryoda CLI çıktısı ile doğrula.

Teslim: Çalışan tek robot simülasyonu, A* kısa yol doğrulaması, seed ile aynı çıktı.

### Faz 2 — Multi-Agent Koordinasyon (MVP’nin Kalbi)
1. `TaskAllocatorAgent`: idle robotlara greedy ETA ile task ataması.
2. `TrafficManagerAgent`: reservation table (`vertex-time`, `edge-time`) + prioritized planning.
3. `RobotAgent`: plan takip, `MOVE/WAIT`, task state geçişleri (to-pickup / to-dropoff / done).
4. `CoordinatorAgent`: tüm ajanları tick bazlı orkestre et.
5. Çakışma mantığı: coordinated modda vertex/edge çakışmaların engellenmesini garanti altına al.
6. Baseline modu ekle: merkezi rezervasyonsuz bağımsız planlama.

Teslim: Aynı senaryoda `baseline` ve `coordinated` çalışır; coordinated modda çakışma yok.

### Faz 3 — Metrikler, Deney Koşucu ve Testler
1. Metrik hesapları: makespan, total path length, task latency, wait count, collision count, (ops.) replanning count.
2. Deney koşucu: senaryo + seed + mode parametreli run ve sonuç üretimi (`json/csv`).
3. En az 2 senaryo için otomatik karşılaştırma script’i.
4. Unit test ve integration test seti.
5. README’ye kurulum/çalıştırma/deney adımları ve beklenen sonuçları ekle.

Teslim: Tek komutla kıyas raporu; testler geçer; README tamam.

### Faz 4 — Minimal Web UI Demo
1. `Streamlit` tabanlı minimal dashboard kur.
2. Senaryo, seed, mode seçim kontrolleri.
3. Grid animasyonu (tick ilerledikçe robotlar + rezervasyon/engeller görünümü).
4. Sağ panelde metrikler ve baseline/coordinated karşılaştırma tablosu.
5. CLI koşucu ile aynı backend kodunu kullan (tek kaynak gerçeklik).

Teslim: Ders demosu için görsel ama hafif bir Web UI.

## 2) Modül/Komponent Taslağı ve Ajanlar Arası Veri Akışı

### Önerilen Dosya Yapısı
- `src/warehouse_sim/models.py`
- `src/warehouse_sim/grid.py`
- `src/warehouse_sim/pathfinding.py`
- `src/warehouse_sim/reservation.py`
- `src/warehouse_sim/agents/robot_agent.py`
- `src/warehouse_sim/agents/task_allocator_agent.py`
- `src/warehouse_sim/agents/traffic_manager_agent.py`
- `src/warehouse_sim/agents/coordinator_agent.py`
- `src/warehouse_sim/policies/baseline_policy.py`
- `src/warehouse_sim/policies/coordinated_policy.py`
- `src/warehouse_sim/metrics.py`
- `src/warehouse_sim/simulator.py`
- `src/warehouse_sim/runner.py`
- `app/dashboard.py`
- `configs/scenarios/*.json`
- `tests/*`

### Sorumluluklar
- `RobotAgent`: local state, mevcut task, plan kuyruğu, `next_action()`.
- `TaskAllocatorAgent`: unassigned task havuzu, `assign_tasks(robots, tasks, grid, tick)`.
- `TrafficManagerAgent`: prioritized path planning, reservation commit/release.
- `CoordinatorAgent`: tick döngüsü, ajan çağrı sırası, state transition, metrik toplama.
- `BaselinePolicy`: rezervasyonsuz bağımsız planlama ve çakışma sayımı.
- `Metrics`: run boyunca event-based sayaçlar ve final agregasyon.

### Tick Veri Akışı
1. Coordinator aktif robot/task durumunu okur.
2. Allocator idle robotlara task atar.
3. TrafficManager coordinated modda öncelik sırasıyla yol planlar ve rezervasyonları işler.
4. Robotlar `MOVE/WAIT` aksiyonunu üretir.
5. Coordinator aksiyonları uygular, task completion kontrol eder.
6. Metrics olayları loglar.
7. Tick artar; tüm tasklar bittiğinde run tamamlanır.

## 3) Senaryo/Konfigürasyon Yaklaşımı

### Konfigürasyon Formatı
`JSON` tek format (ek bağımlılık yok, deterministic parse).  
Ana alanlar:
- `name`
- `seed`
- `grid`: `width`, `height`, `obstacles`
- `stations`: pickup/dropoff hücreleri
- `robots`: id + start
- `tasks`: id + pickup + dropoff + release_tick
- `simulation`: max_ticks, mode
- `events` (opsiyonel, post-MVP): temp_block/delay

### Minimum 2 Senaryo
1. `configs/scenarios/narrow_corridor.json`  
Dar tek şerit koridorda karşılıklı hareket; edge-swap çakışma riskini net gösterir.
2. `configs/scenarios/dense_tasks.json`  
3–6 robot, yoğun görev akışı; assignment + reservation verimini gösterir.

### Determinizm
- Task listesi ve release_tick sabit.
- Tie-breaker’lar sabit sıra ile: `eta`, sonra `robot_id`.
- Aynı seed + aynı mode => aynı sonuç.

## 4) Test Stratejisi (Kritik Hataları Yakalama)

### Unit Testler
1. A* Manhattan kısa yol doğruluğu ve obstacle kaçınma.
2. `WAIT` aksiyonunun geçerli successor olarak işlenmesi.
3. Reservation `vertex-time` çakışma tespiti.
4. Reservation `edge-time` swap çakışma tespiti.
5. Greedy allocator’ın ETA’ya göre doğru robotu seçmesi.
6. Metrics hesaplarının formül doğruluğu.

### Integration Testler
1. Dar koridor senaryosunda coordinated modda `collision_count == 0`.
2. Aynı senaryoda baseline modda `collision_count > 0` veya belirgin conflict event.
3. Aynı seed ile iki koşumda metriklerin birebir eşitliği.
4. Yoğun görev senaryosunda tüm taskların `max_ticks` içinde tamamlanması (coordinated).

### Regression Test Hedefleri
- Deadlock benzeri sonsuz bekleme.
- Task’in iki robota birden atanması.
- Pickup yapılmadan dropoff’a geçiş.
- Reservation temizlenmemesi nedeniyle sahte blokaj birikimi.

## 5) Baseline vs Coordinated Deney Tasarımı

### Protokol
1. Her senaryo için aynı seed setiyle iki modu da çalıştır.
2. Mode A: `baseline` (no reservation).
3. Mode B: `coordinated` (prioritized + vertex/edge reservation).
4. Çıktıları run-bazlı kaydet (`results/*.csv` + `results/*.json`).

### Ölçümler
- makespan
- total path length (move sayısı)
- average task completion time
- wait count
- collision count
- (ops.) replanning count

### Başarı Kriteri Bağlantısı
- Coordinated modda `collision_count = 0` zorunlu.
- Baseline/coordinated aynı senaryoda çalışır ve karşılaştırma tablosu üretir.
- Determinism testi geçer.
- README ve testler tamamdır.

## 6) Opsiyonel Geliştirmeler (MVP Sonrası)
1. Auction tabanlı task allocation (bid = ETA + load penalty).
2. Dinamik olaylar: geçici blokaj/robot gecikmesi.
3. Kısmi replanning: sadece etkilenen robotların yeniden planlanması.
4. Basit belirsizlik modeli: aksiyon başarısızlık olasılığı.
5. Öncelik politikası varyantları: static ID yerine dynamic urgency.
6. UI’de replay timeline ve hız kontrolü.

## 7) Public API / Interface / Type Taslağı
- `run_simulation(config: SimulationConfig, mode: Literal["baseline","coordinated"]) -> RunResult`
- `TaskAllocatorAgent.assign_tasks(state: WorldState) -> list[Assignment]`
- `TrafficManagerAgent.plan_and_reserve(state: WorldState) -> dict[RobotId, Plan]`
- `RobotAgent.next_action(state: LocalObservation) -> Action`
- `MetricsCollector.finalize() -> MetricsReport`
- `load_scenario(path: str) -> SimulationConfig`
- Temel tipler: `Position`, `Task`, `RobotState`, `PlanStep`, `ReservationTable`, `RunResult`, `MetricsReport`.

## Varsayımlar ve Seçilen Varsayılanlar
1. Dil: `Python 3.11+`.
2. UI: `Streamlit` minimal dashboard.
3. Konfigürasyon: JSON.
4. Grid hareketi: 4-neighborhood (diagonal yok).
5. Prioritized planning: static deterministic priority (`eta`, `robot_id`).
6. Horizon: planlama ufku `max_ticks` veya senaryodan türetilen üst sınır.
7. Belirsizlik/replanning: MVP dışı, extension fazında.

## EXECUTE Aşaması Checklist
- [ ] Proje iskeleti ve temel veri modellerini oluştur.
- [ ] A* + Manhattan + WAIT implement et ve unit testlerini yaz.
- [ ] Tick tabanlı simulator çekirdeğini kur.
- [ ] `RobotAgent`, `TaskAllocatorAgent`, `TrafficManagerAgent`, `CoordinatorAgent` sınıflarını ekle.
- [ ] Reservation table (vertex+edge) ve prioritized planning’i entegre et.
- [ ] Baseline policy’yi ekle ve coordinated ile aynı runner’dan çalıştır.
- [ ] Senaryo JSON’larını (`narrow_corridor`, `dense_tasks`) ekle.
- [ ] Metrik toplama ve kıyas raporu üretimini tamamla.
- [ ] Unit + integration testlerini tamamla, determinism testini ekle.
- [ ] README’yi kurulum, çalıştırma, senaryo, kıyas çıktıları ile finalize et.
- [ ] Streamlit dashboard’u bağla ve canlı demo akışını doğrula.
