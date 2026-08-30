# Dynamic Neural Network — Structure-Learning via ML Controller

Experimenteller Ansatz: Ein **externer Machine-Learning-Algorithmus** steuert, *wann* und *wo* ein (weiterhin mit Backpropagation trainiertes) MLP auf MNIST wächst bzw. schrumpft — auf Basis der **Korrelationsmatrix** der verdeckten Aktivierungen.

Kernidee: Statt das Neuronennetz nur anhand von Heuristiken (z. B. mittlere Aktivierung) zu prunen, lernt ein kleiner Regressions-Scorer, die **Wichtigkeit jedes Neurons** vorherzusagen. Er entfernt gezielt unwichtige bzw. redundante Neuronen und schützt Ziffern-spezialisiere. So wird die Modellstruktur selbst Gegenstand des Lernens.

## Hauptidee

- **Korrelationsmatrix** über die Hidden-Aktivierungen → Redundanz / Spezialisierung pro Neuron.
- **NeuronScorer** (kleines MLP `6→64→32→1`): per-neuron Features → vorhergesagte Wichtigkeit.
  - Features (pro Neuron, lokal je Schicht auf `[0,1]` normalisiert): mittlere Aktivierung, Partizipationsrate, mittlere/|max| intra-layer Korrelation, Klassen-Spezifität (z-Wert), Schichtindex.
  - Label (Ground-Truth): **OBD-artige Gradient-Salienz** `|a·∇L|`, je Schicht in einen Rang `0..1` überführt → dadurch **layout-invariant**.
- Trainiert wird der Scorer **meta über mehrere Netzarchitekturen** (verschiedene Tiefe/Breite), sodass eine einzige Instanz über beliebige MLP-Layouts generalisiert (Zero-shot ohne Retraining).

## Dateien

| Datei | Zweck |
|-------|-------|
| `structural_ml_mnist.py` | Platzhalter/Basis-Idee + passives Fallback-Pruning (alte Version) |
| `structural_ml_mnist_v2.py` | Implementierter ML-Kontroller (trainierter Scorer) + acc-bewusstes Gate ("WANN"); Vergleich Baseline vs. ML |
| `structural_ml_metatrain.py` | **Meta-Training** eines layout-invarianten Scorers über mehrere Architekturen + Zero-shot-Test auf ungesehenen Layouts |
| `mnist_mlp.ipynb` | Referenz-MLP (784→256→128→64→10) |
| `models/neuron_scorer_meta.pt` | Der trainierte, layout-invariante Meta-Scorer |
| `scorer_training_curve.png` | Verlauf des Meta-Scorer-Trainings (Loss) |
| `PROJECT_REPORT.md` | Projekt-Audit / aktueller Stand |

**Benchmarks & Vergleiche** liegen gebündelt in [`Report/Benchmarks/`](Report/Benchmarks/README_BENCHMARKS.md):
- `structural_ml_sweep.py` — Pareto-Sweep (Accuracy vs. Synapsen), Baseline vs. ML
- `structural_ml_metascorer_compare.py` — Meta-Scorer (zero-shot) vs. Baseline auf denselben Zielgrößen
- `structural_ml_compact_benchmark.py` — **Laufzeit-Benchmark**: Masken-Pruning vs. physisches Kompaktieren auf GPU **und CPU**; zeigt, dass Kompaktieren (v. a. auf CPU/ESP32-nah ~1.86x) echte Laufzeit spart, Masken-Pruning nicht
- `structural_ml_trainvssmall_benchmark.py` — **Vortrainieren & physisch Prunen** (`784→256→128→64→10` → `[96,48,24]`) vs. direkt gleich groß trainieren (`784→96→48→24→10`); PRUNE schlägt SMALL um ~+0.7 pp bei identischer Größe (81.264 Synapsen)
- `structural_ml_ml_budget_pruning.py` — **ML entscheidet die Schichtgrößen selbst**: nur ein Synapsen-Budget (81.264) ist vorgegeben; der Meta-Scorer + Synapsen-Kosten bestimmen, welche Neuronen/schichten geprunt werden. Ergebnis: überraschend robust `[79,128,64]` (nur die teure erste Schicht wird reduziert, acc steigt sogar leicht)

## Ergebnisse (Kurzfassung)

- **Sweep (8 Zielgrößen, Baseline vs. ML):** ML kontrolliert Pruning hält die Accuracy bei kleinen Zielen deutlich besser (z. B. bei `[96,48,24]`: ML ~86 % vs. Baseline ~74 %).
- **Meta-Training (layout-invariant):** Ein trainierter Scorer generalisiert Zero-shot auf ungesehene Layouts (`[128,128,64]`, `[160,80]`, `[64,64,64,32]`) und übertrifft dort konsistent das Fallback-Pruning.
- **Gleicher-Ziel-Vergleich:** Der Zero-shot-Meta-Scorer gewinnt bei 6 von 7 Prune-Zielgrößen, am deutlichsten bei kleinen Zielen (bis +0.45 pp).

## Setup

- Python 3.12, PyTorch mit CUDA (z. B. `cu126`), torchvision, numpy, matplotlib.
- MNIST wird beim ersten Lauf automatisch heruntergeladen (`torchvision.datasets.MNIST`).
- CUDA (RTX 3070) wird erkannt; ohne GPU läuft alles auf CPU.

```bash
# Meta-Scorer trainieren (5 Train-Layouts, 1500 Epochen) + Zero-shot-Test
python structural_ml_metatrain.py

# Benchmarks & Vergleiche (siehe Report/Benchmarks/README_BENCHMARKS.md)
cd Report/Benchmarks

# Speedup-Benchmark: Masken- vs. Kompaktier-Pruning auf GPU + CPU
python structural_ml_compact_benchmark.py --devices cuda cpu --csv benchmark_compact_GPU_vs_CPU_raw.csv

# Pareto-Sweep über 8 Zielgrößen
python structural_ml_sweep.py

# Vergleiche Meta-Scorer (zero-shot) vs. Baseline auf denselben Zielgrößen
python structural_ml_metascorer_compare.py
```

Hinweis: MNIST-Rohdaten und Trainings-Logdateien sind nicht im Repository enthalten.
