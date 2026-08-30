# Benchmarks & Vergleiche — Dynamic Neural Network

Dieser Ordner enthält **Benchmarks und Vergleichs-Skripte** sowie deren **Ergebnisse** zum
Dynamic-Neural-Network-Projekt (Struktur-Learning via ML-Controller auf MNIST).

> Hinweis: Die Haupt-Trainingsskripte (`structural_ml_mnist_v2.py`, `structural_ml_metatrain.py`,
> `structural_ml_mnist.py`) liegen im Projektordner `/Code`. Hier liegen nur die **Mess-/Vergleichsteile**.

---

## 📁 Dateien im Überblick

### 🔧 Skripte (Benchmark / Vergleich)

| Datei | Inhalt |
|-------|--------|
| `structural_ml_compact_benchmark.py` | **Laufzeit-Benchmark**: Vergleicht Masken-Pruning (Aktivierungen nullen, Matrizen bleiben voll) gegen **physisches Kompaktieren** (Matrizen wirklich schrumpfen). Misst die Forward-Zeit je Batch auf **GPU und CPU**, plus Accuracy und Synapsen-Zahl. Beweis, dass Masken-Pruning keine Laufzeit spart, Kompaktieren aber auf CPU schon. |
| `structural_ml_metascorer_compare.py` | **Gleicher-Ziel-Vergleich**: Trainierter, layout-invarianter **Meta-Scorer (zero-shot)** gegen das Fallback-Pruning (mittlere Aktivierung) auf denselben 8 Zielgrößen. Produziert Plot + CSV. |
| `structural_ml_sweep.py` | **Pareto-Sweep**: Baseline vs. ML-Controller über 8 Zielgrößen; Pareto-Front Accuracy vs. Synapsen. Zeigt, wo ML das Pruning besser kontrolliert. |
| `structural_ml_trainvssmall_benchmark.py` | **"Trainieren & Prunen" vs. "Direkt klein trainieren"**: Vergleicht ein auf `[256,128,64]` trainiertes und physisch auf `[96,48,24]` kompaktiertes Netz gegen ein von Anfang an gleich großes `784→96→48→24→10` Netz. Beide enden bei identischer Größe (81.264 Synapsen). |

### 📊 Ergebnis-Dateien (Rohdaten + Plots)

| Datei | Inhalt |
|-------|--------|
| `benchmark_compact_GPU_vs_CPU_raw.csv` | Rohergebnis des Speedup-Benchmarks (GPU + CPU): Zeiten pro Forward, Speedup-Faktoren, Synapsenzahl, Accuracy. |
| `RESULT_bild_speedup_gpu_cpu.png` | Balkendiagramm: Absolute Forward-Zeit (links) und Speedup-Faktor (rechts) für Masken- vs. Kompaktier-Ansatz auf GPU und CPU. |
| `RESULT_bild_sweep_accuracy_vs_synapses.png` | Pareto-Graph des Sweeps: Accuracy über Synapsen, Baseline vs. ML-Controller. |
| `RESULT_bild_sweep_metascorer_compare.png` | Vergleichsplot Meta-Scorer (zero-shot) vs. Baseline über die Zielgrößen. |
| `RESULT_bild_zero_shot_metascorer.png` | Zero-shot-Ergebnis des Meta-Scorers auf ungesehenen Layouts. |
| `RESULT_sweep_metascorer_compare.csv` | Rohdaten zum Meta-Scorer-Vergleich (Zielgrößen, Accuracy Baseline vs. Meta). |
| `RESULT_zero_shot_metascorer.csv` | Rohdaten zum Zero-shot-Ergebnis (Layout, Accuracy Baseline vs. Meta). |
| `RESULT_trainvssmall.csv` | Rohdaten des Trainieren&Prunen-vs.-klein-Benchmarks (Acc je Variante, Synapsen). |
| `RESULT_bild_trainvssmall_acc.png` | Balkendiagramm: Accuracy von VOLL, PRUNE (kompaktiert) und SMALL (gleich groß). |

---

## ⚡ Ergebnis des Laufzeit-Benchmarks (GPU vs. CPU)

Fragestellung: *"Bringt das Pruning eine echte Laufzeitverbesserung?"*

**Antwort:** Nein beim **Masken-Pruning** (Aktivierungen nullen, Matrizen bleiben voll — die
Matrixmultiplikation läuft weiter über alle Kanäle). **Ja beim physischen Kompaktieren** —
insbesondere auf CPU (dem ESP32-nahen Fall).

Messbedingungen: MNIST-MLP `784→256→128→64→10`, Prune-Ziel `[96,48,24]`, Batch = 1024,
200 (GPU) bzw. 500 (CPU) Forward-Durchläufe. `Syn` = Anzahl Synapsen (Mac-Multiplikationen).

| Gerät | Ansatz | Zeit (ms/Fwd) | Accuracy | Syn | Speedup | Anmerkung |
|-------|--------|---------------|----------|-----|---------|-----------|
| **GPU** (RTX 3070) | Volles Netz | 0.444 | 97.14% | 242 304 | 1.0x | Referenz |
| **GPU** | Masken-Pruning | 0.346 | 43.74% | 242 304 | 1.28x | Matrizen voll → kein realer Gewinn |
| **GPU** | Physisch kompakt | 0.501 | 43.74% | 81 264 | 0.89x | kleiner + schrumpfende Breite; GPU-Overhead dominiert |
| **CPU** | Volles Netz | 1.273 | 97.17% | 242 304 | 1.0x | Referenz |
| **CPU** | Masken-Pruning | 1.355 | 42.24% | 242 304 | 0.94x | ≈ 1.0x → kein Speedup |
| **CPU** | Physisch kompakt | 0.683 | 42.24% | 81 264 | **1.86x** | echter, großer Gewinn |

### Interpretation

- **Synapsen-Reduktion:** 66.5 % (242 304 → 81 264) durch Kompaktieren.
- **GPU:** Kein nennenswerter Gewinn (0.89–1.28x). Grund: Kernel-Launch-/Parallelitäts-Overhead
  dominiert bei so kleinen Netzen; zudem bleibt die dominante Eingangsschicht `784→256`
  (83 % der Synapsen) beim Pruning unverändert (die 784 Pixel-Eingänge sind fix).
- **CPU (≈ ESP32-Charakteristik):** Physisches Kompaktieren ist **~1.86x schneller** —
  dort ist die Rechenzeit proportional zur Synapsenzahl, Kern-Overhead spielt kaum eine Rolle.
  Masken-Pruning bleibt nutzlos (~0.94x).
- **`masked == compact` (assert bestätigt):** Nach dem Kompaktieren ist die Mathematik
  identisch zur Masken-Variante — es ist also *dasselbe Modell*, nur ohne totes Gewicht.

### Konsequenz für den ESP32

Auf dem ESP32 (ARM-Cortex-CPU, kein CUDA) gilt die **CPU**-Zeile: Das **physische
Kompaktieren** lohnt sich real (~doppelte Geschwindigkeit), nicht aber das bloße
Nullen von Aktivierungen. Für maximale Geschwindigkeit wären zusätzlich **Quantisierung
(int8)** und ein mitbedachtes Schrumpfen der Eingangsschicht anzuraten (die 784 Pixel
dominiert die Matrizen und schrumpft beim Neuronen-Pruning nicht).

---

## 🎯 Ergebnis: "Trainieren & physisch Prunen" vs. "Direkt klein trainieren"

Fragestellung: *"Ist es besser, das volle Netz zu trainieren und dann physisch zu
kompaktieren, oder von Anfang an ein gleich großes schlankes Netz zu trainieren?"*

Messbedingungen: MNIST, Seeds `[42, 2024]` (Mittelwert). „PRUNE“ = `784→256→128→64→10`
mit 6 Epochen trainieren → physisch auf `[96,48,24]` kompaktieren → 6 Epochen Fine-Tune.
„SMALL“ = `784→96→48→24→10` direkt mit 12 Epochen trainieren (gleiche Trainingssumme).
„VOLL“ = `784→256→128→64→10` mit 12 Epochen als Referenz.

| Variante | Architektur am Ende | Acc (%) | Syn/MACs |
|----------|---------------------|---------|----------|
| **VOLL**  | `784→256→128→64→10` | 98.04 | 242 304 |
| **PRUNE** | `784→96→48→24→10` (kompaktiert) | **97.83** | **81 264** |
| **SMALL** | `784→96→48→24→10` (direkt) | 97.11 | **81 264** |

### Interpretation

- **PRUNE und SMALL enden bei identischer Größe** (81.264 Synapsen, −66.5 % vs. VOLL).
- **PRUNE schlägt SMALL um +0.71 pp** (97.83 % vs. 97.11 %): Das auf dem vollen Netz
  trainierte und danach kompaktierte Netz ist genauer als ein von Anfang an gleich
  großes Netz — das Vortraining auf der größeren Architektur verleiht einen Vorteil.
- Gegenüber VOLL verliert PRUNE nur **−0.21 pp**, spart aber **66.5 % Synapsen** —
  ein sehr gutes Verhältnis aus Genauigkeit und Größe.
- **Fazit für dein ESP32-Szenario:** Der geplante Ablauf (auf dem PC vortrainieren →
  physisch kompaktieren → Forward-Pass auf dem Board) ist nicht nur machbar, sondern
  liefert sogar ein etwas besseres Modell als eine direkt klein trainierte Alternative —
  bei gleichem Speicher- und Rechenbudget.

---

## 🚀 Ausführen

```bash
cd "D:\Dynamic Neural Networt (DNN)\Report\Benchmarks"

# Speedup-Benchmark (GPU + CPU) mit CSV-Ergebnis
python structural_ml_compact_benchmark.py --devices cuda cpu --csv benchmark_compact_GPU_vs_CPU_raw.csv

# Vergleich Meta-Scorer (zero-shot) vs. Baseline
python structural_ml_metascorer_compare.py

# Pareto-Sweep Baseline vs. ML-Controller
python structural_ml_sweep.py

# Trainieren&Prunen vs. Direkt-klein-trainieren (gleiche Zielgroesse)
python structural_ml_trainvssmall_benchmark.py 42 2024
```

> Hinweis: `structural_ml_metascorer_compare.py` lädt den trainierten Meta-Scorer
> (`neuron_scorer_meta.pt`). Das Skript sucht den Pfad automatisch und fällt auf
> `/Code/models/neuron_scorer_meta.pt` zurück, wenn er hier nicht liegt.

---

## Verwandt

- Hauptprojekt & Training: siehe `/Code` (README im Repo-Root).
- MNIST-Rohdaten, Logdateien und Trainings-Intermediates sind nicht versioniert.
