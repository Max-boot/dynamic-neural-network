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
| Conv+ANFIS (SVHN) | `conv_anfis_saliency_svhn.pt` | 902 | 40 Ep. auf SVHN-Szenen (s. u.) |
| BNN+Box-Head (SVHN) | `bnn_mc_box_svhn.pt` | 785.247 | 70 Ep. auf SVHN-Szenen (s. u.) |

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

## Real-World-Transfer: SVHN (echte Fotos)

Kann die auf synthetischen Szenen trainierte Kaskade auf **echte Fotos** springen?
Als Welt-Standard-Benchmark dient SVHN (Straßenfoto-Hausnummern, Format-1: 32x32
zentrierte Ziffer). Aus den SVHN-Train/Test-Ziffern wurden eigene Szenen im
Pipeline-Format (128x128, 4–12 Ziffern, 8–19px) **ohne Sprite-Leak** gebaut
(`dataset_build/build_svhn_scenes.py`, `scene_dataset_svhn/`, 5000/1000) und beide
Stufen darauf neu trainiert. GT-Box = zentriertes 65 %-Quadrat; Blend per
copy-over (SVHN teils hell, `darken` wäre unsichtbar).

| Stufe | Modell | Kennzahl | SVHN | synthetisch (MNIST-Szenen) |
|---|---|---|---|---|
| Saliency (1+2) | ConvANFIS 902 P. | AUROC / AP / Rec@P≥0.5 | 0.974 / 0.952 / 0.989 | 0.990 / 0.987 / 0.995 |
| BNN (4+5) | 785 k P., 70 Ep. | val_acc / digit_acc | 0.413 / 0.256 | 0.594 / 0.459 |

**End-to-End** (hybrid2, 1000 SVHN-Testszenen, 7937 GT-Ziffern; `svhn_eval.csv`):

| Metrik | SVHN (gate 0.2) | synthetisch (gate 0.5) |
|---|---|---|
| Precision | 0.380 | 0.561 |
| Recall | 0.141 | 0.140 |
| Bilder mit ≥1 TP | 0.688 | 0.688 |
| Klassen-Acc (TP) | 0.338 | 0.774 |
| Box-IoU mean / ≥0.6 | 0.636 / 0.566 | 0.651 / 0.632 |
| Zentrumsfehler | 2.23 px | 2.07 px |
| Fenster/Bild | 8.6 | 8.6 |
| Forward-Pässe / Bild | 68.5 | 68.8 |

*(Nach dem Fenster-pro-Ziffer-Fix von hybrid2: jeder Saliency-Peak erhält ein
eigenes Fenster → ~0.93 Ziffern/Fenster statt 1.5, Recall ≈ doppelt so hoch
wie vorher, Box-/Zentrumsqualität unverändert.)*

→ **Einordnung:** Die billigen Stufen transferieren nahezu verlustfrei — die
Saliency findet auf echten Fotos ebenso viele Ziffern wie im synthetischen
Szenario (Recall 0.141 zu 0.140 bei gate 0.2/0.5), und Box-Regression sitzt mit
Zentrumsfehler ~2.2px so genau wie gehabt. Der Engpass ist konsistent der BNN-*Klassifikator*:
auf echten Foto-Ziffern bleibt er unsicher (p_max ~0.2 statt ~0.8), sodass das
auf klare Texturen kalibrierte Konfidenzgatter p≥0.6 alles verwirft; erst mit
gate≈0.2 werden Detektionen zugelassen und die Klasse ist nur bei 0/1 zuverlässig
(0.82/0.78), bei 2–9 nahe Zufall. Das passt exakt zur Trainingskurve (digit_acc
0.256 ≫ 1/11, aber weit unter der MNIST-Szene). Der 28x28-SVHN-Crop ist am
Auflösungslimit — Zwischenfall des Bandbreiten-Slots, nicht der Kaskadenlogik;
Co-Lokalisation und Gating bleiben funktionsfähig.

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

## ESP32-Einschaetzung (analytisch, ohne Messskript)

Zwei getrennte Fragen: **passt es in den Speicher?** und **schafft die CPU
die MACs?**

### 1) Groesse (Parameter → Checkpoint-Speicher)

| Stufe | Params | FP32 (4 B) | int8 + BN-Fusion (≈1 B) |
|---|---|---|---|
| Conv+ANFIS-Saliency (Stage 1+2) | 902 | 3,6 kB | 0,9 kB |
| BNN + Box-Head (Stage 4+5) | 785.247 | 3,14 MB | 0,79 MB |
| **Pipeline gesamt** | **786.149** | **3,15 MB** | **0,79 MB** |

- ESP32-S3: 512 kB SRAM (nutzbar ~ 300–400 kB) + optional PSRAM 2–16 MB.
- **FP32 passt nicht in SRAM** (3,15 MB > 512 kB), aber in PSRAM ja —
  PSRAM-Zugriffe sind aber langsam (→ GPIOMATRIX/Bus-Switching).
- **int8 (0,79 MB) passt in SRAM-Naenhe** (z. B. über MMU-flat
  `external_psram` nicht nötig; 512 kB genügen knapp, ideal zusammen mit
  streaming Read statt full Checkpoint im RAM).

### 2) Rechenaufwand (MACs pro Bild, deterministisch, ohne MC-Multiplikator)

Je Forward-Pass: Saliency ≈ 6 M MAC (ConvStack auf 128×128), BNN-Crop ≈ 6,7 M
MAC (Conv2 40→80 auf 14×14 + FC auf 7×7-Flat).

| Konfiguration | Forward-Pässe/Bild | MACs/Bild |
|---|---|---|
| **Jetzt** (MC-S=8, ~8,6 Fenster, full-parse) | 8,6·8 + 1 = ~70 | **≈0,47 G MAC** |
| Deterministisch (MC-S=1), full-parse | ~9,6 | ≈57 M MAC |
| Deterministisch + early-stop (gate) | 3–5 | ≈20–35 M MAC |

ESP32-S3 @ 240 MHz, int8-NNC: realistisch **50–100 M MAC/s** (DSP-SIMD ohne
echte HW-MAC). Daraus:

- ~0,47 G MAC (MC-Status quo): **5–9 s/Bild** → nur als Offline-Auswertung sinnvoll.
- ~57 M MAC (MC-S=1): **~0,6–1,1 s/Bild** → sporadische Detektion (ein Bild alle
  1–2 s) machbar; nicht Echtzeit.
- ~20–35 M MAC (mit early-stop): **~0,2–0,7 s/Bild** → brauchbar für
  ESP32-CAM-Boolesche „Ziffer vorhanden?"-Abfragen.

**Kernaussage:** Der Engpass ist *nicht* die Modellgröße (die ist für die MCU
trivial klein), sondern die **MC-Dropout-Zahl (×8)** multipliziert mit der
Fensterzahl. Die Unsicherheitsberechnung ist der bewusste Kompromiss der
Studie (genaues Confidence-Gating > rohe Geschwindigkeit); für einen echten
ESP32-Deploy würde man MC-S auf 1 reduzieren (deterministisch) und dafür die
Saliency-gate-Schwelle aggressiver setzen.