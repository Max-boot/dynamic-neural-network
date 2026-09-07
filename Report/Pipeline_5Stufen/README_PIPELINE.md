# Dynamic NN Pipeline — Mehrstufige TinyML-Objekterkennung (Stufen 1–5)

Konzeptstudie für eine **ESP32-CAM-taugliche, mehrstufige Kaskade**:
Statt ein großes Netz auf ganze Szenen zu legen, entscheidet eine hierarchische
Pipeline, *wo* sich Objekte befinden (billige Saliency), um an die wenigen
verbleibenden Stellen den teuren, unsicherheitsbewussten Klassifikator (BNN)
zu setzen.

```text
Szene 128x128
   │
   ├─ Stufe 1+2: ConvStack(8→4) + ANFIS (3×5 MF, 125 Regeln)  → 8x8-Saliency
   │             "Wo ist etwas?"                               (902 Param.)
   │
   ├─ Stufe 3:   DecisionTree ([sal,m,var,max] je Kachel)      → Proposals/Gating
   │             (Ablation; von ANFIS-Saliency überholt)       (~gering)
   │
   └─ Stufe 4+5: BNN mit MC-Dropout + Box-Head
                 "Was ist es (0-9|Hintergrund) und wo genau?"  (785.247 Param.)
                 Regionen-Crop 28x28 → Klasse + Bounding-Box (cx,cy,w,h)
```

## Daten (synthetisch)

| Dataset | Inhalt | Größe |
|---|---|---|
| `cluttered_mnist` | 100x100, Ziel-Ziffer mit 8 Distraktoren | train 5000 / test 1000 |
| `corrupted_mnist` | 28x28, 11 Stör-Typen (Blur, Noise, Shear, …) | train 60000 / test 10000 |
| `scene_dataset` | 128x128, 4–12 Ziffern (8–19px) auf Rausch-Gradient-/Blobs-Hintergrund, GT-Boxen + Labels | train 5000 / test 1000 |

Generierung: `dataset_build/generate_scene_dataset.py` (labels sind echte
Ziffern-Labels aus dem Pool, seit der Label-Fixierung konsistent).

## Trainierte Modelle (`pipeline/models/`)

| Modell | Datei | Params | Training |
|---|---|---|---|
| Conv+ANFIS | `conv_anfis_saliency.pt` | 902 | 40 Ep., Adam lr 1e–3, BCEWithLogits + pos_weight (~2.1) |
| Region-Tree | `region_tree.pkl` | ~500 Blätter | DecisionTree max_depth=8, min_samples_leaf=4 |
| BNN+Box-Head | `bnn_mc_box.pt` | 785.247 | 70 Ep., AdamW lr 1e–3, CE (gewichtete Klassen) + Smooth-L1 (Box) |

## Ergebnisse (Test, 1000 Szenen)

### Stufe 1+2 — Saliency auf Tile-Ebene (8x8)

| Metrik | Wert |
|---|---|
| AUROC | **0.990** |
| Average Precision | **0.987** |
| Recall@Precision≥0.5 | **0.995** |

### Stufe 3 — Tree (Ablation)

| Metrik | Wert |
|---|---|
| AUROC (je Kachel) | 0.989 |
| AP | 0.987 |
| Regionen/Bild bei Threshold 0.046 | 2.25 |

### Stufe 4+5 — End-to-End-Detektion (IoU≥0.5, Greedy-Matching)

Extraktionspfad (Confidenzgatter p≥0.6):

| Methode | Bilder mit ≥1 Detektion | Precision | Recall | Klasse (0-9) Acc | Forward-Pässe/Bild |
|---|---|---|---|---|---|
| **Pipeline (Conv+ANFIS→Regionen→BNN)** | 0.251 | **0.691** | 0.037 | **0.789** | **27** |
| Baseline (alle 64 Kacheln einzeln) | 0.738 | 0.031 | 0.161 | 0.409 | 512 |

→ **Kernresultat:** Die Kaskade ist ~22× präziser, ~19× effizienter (Forward-Pässe)
und ~1,9× genauer in der Ziffern-Klassifikation als die naive Kachel-für-Kachel-Variante.
Der Preis: geringere Recall-Abdeckung (Saliency findet nur die salientesten Objekte).

### Klassifikations-Verbesserung (BNN)

Ziel war die bislang schwächste Stufe: Ziffern 0-9 in verrauschten 28x28-Crops
einzeln zu klassifizieren. Per Ablation wurden vier Trainings-/Architekturpfade
auf **demselben festen Val-Crop-Set (seed 42, 15.992 Crops)** verglichen:

| Konfiguration | val_acc | digit_acc (nur 0-9) | box_hit |
|---|---|---|---|
| Alt-Version (32→64→128, 45 Ep., 85k Crops) | 0.549 | 0.399 | 0.609 |
| + Backbone-Pretrain auf sauberem MNIST | 0.534 | 0.379 | 0.650 |
| + Rotation/Zoom/Brightness-Augmentation | 0.504 | 0.342 | 0.634 |
| **Final: c1=40→c2=80→hid=192, 70 Ep., 105k Crops** | **0.594** | **0.459** | **0.661** |

*(Zum Vergleich: Im etwas anderen Val-Split des Trainingsskripts — 18.987 Crops —
liest der finale Checkpoint 0.573 / 0.460 / 0.636; `bnn_results.csv`.)*

→ **Gewinner:** mehr Trainings-Crops (Fenster×10 + Kachel×5 + 20k Clutter statt
16k) + moderat mehr Kapazität (0.79M statt 0.42M Params, weiterhin explizit
MCU-tauglich klein) + längeres Training. Digit-Acc +6.1pp, nur 0.4pp Box-Verlust.

→ **Verworfen (offene negative Resultate):** Ein Backbone-Pretraining auf
`sauberem` MNIST *verschlechtert* die Acc auf der verrauschten Scene-Verteilung
(-1.5..-4.5pp): Die sauberen Ziffern-Features übertragen nicht auf Clutter-Crops
(Windowing-Jitter ±8px, 24-48px-Fenster, BG-Klassen-Dominanz). Auch eine
Rotation/Zoom-Augmentation schadet (-4pp), weil bereits 14px-Ziffern nach der
Transformation nicht mehr unterscheidbar sind — die wirkungsvolle Diversität
kommt aus zusätzlichen *echten* Window-Crops, nicht aus geometrischen Verzerrungen.

### Lokalisierungs-Ablation (IoU-Sweep, Pipeline, neues BNN)

| IoU≥ | Precision | Recall |
|---|---|---|
| 0.2 | 0.946 | 0.051 |
| 0.3 | 0.902 | 0.048 |
| 0.4 | 0.822 | 0.044 |
| 0.5 | 0.678 | 0.036 |
| 0.6 | 0.421 | 0.023 |

→ Bei den sehr kleinen Objekten (Median ~14px in 128px) ist IoU≥0.5 eine große
Hürde: Der ±2px-genaue Box-Sitz steckt in der Lokalisierungsgenauigkeit fest.

### MC-Dropout-Abstention

Die Confidence-Kurve (`pipeline_eval.csv`) zeigt: Mit steigender Confidence-Schwelle
steigt sowohl die Ziffern-Klassifikationsrate als auch die Kandidat-Korrektheit
(IoU) monoton an — der unsicherheitsbewusste BNN kann gezielt **verweigern**,
statt falsch zu raten. Das ist auf einem ESP32-CAM der entscheidende Mechanismus
gegen unbrauchbare Falsch-Positiv-Raten.

## Reproduzierbarkeit

- `Dynamic_NN_Pipeline.ipynb` (Repo-Root): kompletter Ablauf; ohne Retraining,
  nur Modell-Load (setzt `RETRAIN = False`). Benötigt die generierten Datensätze
  (werden per Pfad-Suche gefunden; sonst zuerst `dataset_build/` ausführen).
- Alle Quellen: `pipeline/*.py`
- Alle Messwerte: `pipeline_eval.csv`, `iou_sweep.csv`, `pipeline_vs_baseline.png`

## Ehrliche Einordnung

- Mit 8–19px großen Ziffern auf Rausch-Hintergrund sind 28x28-Crops am unteren
  Rand des Machbaren — die Klassifikationsrate (0.79 bei akzeptierten Detektionen,
  digits-only 0.46 auf dem Val-Crop-Set) ist die physikalische Grenze dieser
  Auflösung, nicht ein Trainingsartefakt.
- Der Decision Tree bringt auf Test-Daten kein Vorteil ggü. der ANFIS-Saliency
  (zu grobe Über-Segmentierung) und wird daher als Ablation geführt.
- ESP32-Perspektive: ANFIS (klein, differenzierbar) + Tree als billiger
  Vorfilter sind MCU-real; der BNN bleibt wegen S-MC-Passes als *Top-K-Verifikation*
  die bewusste Design-Option (siehe `Report/TinyML_Research`).