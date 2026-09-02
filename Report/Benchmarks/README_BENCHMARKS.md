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
| `structural_ml_ml_budget_pruning.py` | **ML entscheidet die Schichtgrößen selbst**: Statt festem Ziel `[96,48,24]` wird nur ein **Synapsen-Budget** (81.264) vorgegeben. Der Meta-Scorer bewertet jedes Neuron, und das System prunt schicht-übergreifend nach Wichtigkeit **und** Synapsen-Kosten (teure frühe Neuronen zuerst), bis das Budget erreicht ist. Die resultierenden Schichtgrößen = das "Verhalten" des ML. |
| `structural_ml_prune_methods_compare.py` | **Vergleich der Wichtigkeits-Metriken (gleiches Budget 40k)**: `ML-Score × Ersparnis` vs. `ML-Score × √Ersparnis` vs. `L1-Norm` (TinyML-Standard). Zeigt: L1-Norm schlägt den trainierten Scorer deutlich. |

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
| `RESULT_ml_budget_pruning.csv` | Rohdaten des ML-Budget-Prunings (Acc vor/nach, Synapsen, vom ML gewählte Schichtgrößen). |
| `RESULT_bild_ml_budget_pruning.png` | Verlauf des Synapsen-Abbaus + vom ML gewählte Schichtgrößen (vor/nach). |
| `RESULT_prune_methods_compare.csv` | Rohdaten des Wichtigkeits-Metrik-Vergleichs (Acc je Methode/Seed, Synapsen, Schichtgrößen). |
| `RESULT_bild_prune_methods.png` | Balkendiagramm: Acc der drei Prune-Methoden (ML×s, ML×√s, L1) nach Budget-Pruning. |

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

## 🧠 Ergebnis: "ML entscheidet die Schichtgrößen selbst" (Budget-Pruning)

Fragestellung: *"Was passiert, wenn nicht fest vorgegeben wird, wie viele Neuronen
je Schicht übrig bleiben — sondern nur ein Synapsen-Budget, und das ML selbst
entscheidet, WELCHE Neuronen in WELCHER Schicht geprunt werden?"*

Regel: Der Meta-Scorer bewertet jedes Neuron (0..1 Wichtigkeit). Geprunt wird iterativ
das Neuron mit **`prio = (1 − score) × Synapsen-Ersparnis`** — also zuerst die
**teuren (frühen) und zugleich unwichtigen** Neuronen, weil sie das Budget am
effizientesten erreichen. Ein Floor pro Schicht (15 %, mind. 8) verhindert den
Kollaps einer Schicht. Budget = 81.264 Synapsen (≈ −66.5 % von 242.304).

| Seed | VOLL (vorher) | vom ML gewählt (nachher) | Synapsen | Accuracy vorher → nachher |
|------|---------------|--------------------------|----------|---------------------------|
| 42   | `[256,128,64]` | `[79,128,64]` | 242 304 → 80 880 (−66.6 %) | 97.38 % → **97.57 %** |
| 2024 | `[256,128,64]` | `[79,128,64]` | 242 304 → 80 880 (−66.6 %) | 97.57 % → **97.69 %** |

### Interpretation (das "Verhalten" des ML)

- **Beide Seeds wählen identisch `[79,128,64]`** → sehr robuste, konsistente
  Entscheidung, keine Zufallswahl.
- Das ML schrumpft **nur die erste (teuerste) Schicht** (256→79) und lässt die
  hinteren beiden (128, 64) **völlig unberührt**. Jedes Neuron in Schicht 1 spart
  784+128 = **912 Synapsen**; das ML erkennt, dass die Eingangsschicht stark
  überdimensioniert ist und am effizientesten zu reduzieren ist, während die
  späteren Schichten pro Synapse wichtiger sind.
- Die Accuracy **steigt sogar leicht** (≈ +0.2 pp trotz −66.6 % Synapsen).
- **Kontrast zur festen Vorgabe `[96,48,24]`:** Dort wurde proportional in allen
  Schichten geprunt. Das ML-Budget-Pruning findet ein **unausgeglicheneres,
  kostenoptimiertes** Layout — es konzentriert die Einsparung auf die Eingangsschicht.

> Transparenz-Hinweis: Die Ersparnis-Wichtung (`× Ersparnis`) ist eine bewusste
> Design-Entscheidung ("kostenbewusstes ML-Pruning"). Ohne diesen Faktor (nur nach
> Score) würde das Layout eher proportional zu den Originalgrößen ausfallen.

### Wie ändert sich das ML-Verhalten mit dem Budget? (40.000 Synapsen)

Dasselbe Skript mit aggressiverem Budget `SYN_TARGET = 40000` (~ −83.5 % statt
−66.5 %):

| Seed | VOLL (vorher) | vom ML gewählt (nachher) | Synapsen | Accuracy vorher → nachher |
|------|---------------|--------------------------|----------|---------------------------|
| 42   | `[256,128,64]` | `[38,122,42]` | 242 304 → 39 972 | 97.38 % → **97.24 %** |
| 2024 | `[256,128,64]` | `[38,113,48]` | 242 304 → 39 990 | 97.57 % → **97.00 %** |
| Mittel | `[256,128,64]` | `[38,118,45]` | ~40 000 | ≈ 97.1 % |

Interpretation:
- Bei diesem viel knapperen Budget reicht das Schrumpfen **nur** der
  Eingangsschicht nicht mehr. Das ML prunt Schicht 1 **bis auf den Floor (38)**
  und greift nun **zusätzlich auf die späteren Schichten** zurück (Schicht 2 → 118,
  Schicht 3 → 45).
- Der Floor-Mechanismus (15 % der Originalgröße, min. 8) verhindert, dass Schicht 1
  noch weiter kollabiert, und zwingt das ML, die wenigen verbliebenen Synapsen
  effizient über alle Schichten zu verteilen.
- Accuracy bleibt trotz −83.5 % Synapsen hoch (≈ 97.1 %): massiv komprimiert, aber
  nahezu gleich gut.
- **Fazit für das "Verhalten" des ML:** Das gewählte Layout hängt klar vom Budget ab.
  Bei moderater Kompression (81k) schrumpft es fast nur die teure erste Schicht,
  bei starker Kompression (40k) muss es zusätzlich alle Schichten proportionaler
  beschneiden.

---

## 📊 Vergleich der Wichtigkeits-Metriken (Budget 40.000)

Drei Methoden, **identische Bedingungen** (gleiches Netz `[256,128,64]`, Budget
40.000, gleicher Floor, gleiche Seeds 42/2024, gleiche Warm-/Fine-Tune-Epochen).
Nur die *Wichtigkeits-Metrik* unterscheidet sich — alle prunen greedy bis ans Budget:
`prio = (1 − score) × sav^p`:

| Methode | save-Potenz | Synapsen | Acc voll | Acc nach | Delta |
|---|---|---|---|---|---|
| A: ML-Score × Ersparnis | 1 | 39 892 | 97.47 % | 96.69 % | **−0.78** |
| B: ML-Score × √Ersparnis | 1/2 | 39 835 | 97.47 % | 96.72 % | **−0.76** |
| C: **L1-Norm** × Ersparnis | 1 | 39 900 | 97.47 % | **97.38 %** | **−0.09** |

### Interpretation

- **L1-Norm (TinyML-Standard) schlägt den trainierten Meta-Scorer deutlich**
  (−0.09 pp statt ~−0.77 pp Verlust). Bei Seed 42 übertrifft L1 sogar das
  ungedruckte Netz (97.44 % vs. 97.32 %).
- **√Ersparnis bringt praktisch nichts** (−0.76 vs. −0.78) — der Ersparnis-Bias ist
  nicht die Ursache des Accuracy-Verlusts der Score-Methoden.
- **Warum gewinnt L1?** Der Meta-Scorer ist *layout-invariant* trainiert und liefert
  die Wichtigkeit als **Rang innerhalb einer Schicht** (relativ zu den
  Schicht-Kameraden). Er vergleicht Schicht A **nicht absolut** gegen Schicht B.
  Die **L1-Norm ist ein absolutes, globales** Maß über alle Schichten (reale
  Gewichtsmagnitude). Beim aggressiven 40k-Budget zählt die *absolute* Wichtigkeit —
  und da ist die absolute L1-Norm dem relativen Rang-Score überlegen.
- L1 wählt fast immer `[38,128,38]` (Schicht 1 und 3 stark, Schicht 2 voll erhalten),
  während die Score-Methoden anders über Schichten verteilen.

> Einordnung: Der komplexe (trainierte, RL- etc.) Pruning-Ansatz ist hier NICHT
> besser als die einfache L1-Norm. Für MCU-/Edge-Deployment (wo die Industrie
> ohnehin auf L1-Magnitude setzt) ist die einfache Metrik bei starkem Budget die
> robustere Wahl.

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

# ML entscheidet die Schichtgroessen selbst (Synapsen-Budget statt festem Ziel)
python structural_ml_ml_budget_pruning.py 42 2024

# Vergleich der Wichtigkeits-Metriken: ML-Score x s  vs  ML-Score x sqrt(s)  vs  L1-Norm
python structural_ml_prune_methods_compare.py
```

> Hinweis: `structural_ml_metascorer_compare.py` lädt den trainierten Meta-Scorer
> (`neuron_scorer_meta.pt`). Das Skript sucht den Pfad automatisch und fällt auf
> `/Code/models/neuron_scorer_meta.pt` zurück, wenn er hier nicht liegt.

---

## Verwandt

- Hauptprojekt & Training: siehe `/Code` (README im Repo-Root).
- MNIST-Rohdaten, Logdateien und Trainings-Intermediates sind nicht versioniert.
