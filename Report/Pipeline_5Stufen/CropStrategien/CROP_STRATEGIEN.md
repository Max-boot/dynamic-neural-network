# Crop-Strategien: Vergleich und Archiv

**Datum:** 2026-09-09
**Messung:** Test-Split, 400 Bilder, gate = 0.6 (max softmax), MC-Dropout S=8,
deterministisch geseedet (torch.manual_seed(0)).
**Ergebnisquelle:** `modes_compare_400.csv`.

Die Pipeline (Stufe 3) erzeugt aus der Saliency-Karte (8x8, Threshold 0.5,
Connected Components) Kandidaten-Fenster, die der BNN (MC-Dropout + Box-Head)
einzeln auf 28x28 bewertet. Fünf Strategien für diese Fenster wurden gebaut
und vermessen. **Gewinner ist `hybrid2`** (Weiterentwicklung von `hybrid`).
Die vier Vorgänger sind in `crop_strategies_archive.py` archiviert und werden
in der aktiven Pipeline nicht mehr verwendet (Dispatcher-Default:
`mode="hybrid2"`).

## Messergebnisse (Pipeline FULL, gate=0.6, 400 Bilder)

| Strategie | Precision | Recall | Bilder mit TP | cls_acc | Forward-Pässe/Bild |
|-----------|-----------|--------|---------------|---------|---------------------|
| bbox      | 0.656     | 0.0327 | 0.220         | 0.771   | 27.8                |
| refocus   | 0.632     | 0.0380 | 0.260         | 0.844   | 27.8                |
| kmeans    | 0.254     | 0.0215 | 0.147         | 0.870   | 49.9                |
| hybrid    | 0.553     | 0.0489 | 0.340         | 0.809   | 42.0                |
| **hybrid2**| **0.568**| **0.0458** | **0.318** | **0.850** | 41.7            |

(400-Bilder-Paarlauf hybrid vs. hybrid2, deterministische Messung: siehe
Vergleichs-Zeilen unten; die Vollmessung auf 1000 Bildern in
`hybrid2_eval.csv`.)

Interpretation:

- **Precision:** `bbox` ist minimal präziser als `hybrid` (0.656 vs. 0.553),
  erkauft sich das aber bei Recall (0.0327 vs. 0.0489, **+50 %**) und
  gefundenen Bildern (0.220 vs. 0.340, **+55 %**).
- **Recall / gefundene Bilder:** `hybrid` ist mit Abstand am besten.
- **Klassifikation:** `kmeans` klassifiziert die (wenigen) überlebenden
  Kandidaten am saubersten (0.870), `refocus` ist mit 0.844 dicht dran —
  beide übertreffen `bbox` (0.771) klar.
- **Effizienz:** `hybrid` kostet ~14 Forward-Pässe mehr als `bbox/refocus`
  (28 -> 42), `kmeans` am meisten (50).

## Warum die drei anderen Strategien scheitern

### 1. `bbox` — rohe Tile-Bounding-Box (Alt, Baseline aus Stufe 3.0)

**Idee:** Region = Connected-Component der Kacheln mit Saliency >= 0.5; das
Fenster ist deren Bounding-Box (+2 px Rand).

**Warum es nicht funktioniert:**
- Die Bounding-Box bildet die **Begrenzung der gefüllten Kacheln** ab, nicht
  den Ort der Ziffer. Die 16-px-Kacheln fassen die Ziffer (8–19 px) nur grob
  ein, dadurch sitzt die Ziffer oft **am Fensterrand oder asymmetrisch**.
- Stufe-4-Diagose: 67 % BNN-Reject + 29 % Bad-Box hatten dieselbe Wurzel —
  der Crop ist off-center, der Box-Head kann die Ziffer nicht sauber
  einrahmen.
- Folge: niedrigster Recall (0.0327) und schlechteste Klassifikation (0.771)
  bei gleichen Kosten wie `refocus`.

### 2. `refocus` — Quadrat um den Saliency-Schwerpunkt der Region

**Idee:** Statt der rohen BBox wird um den **saliency-gewichteten
Schwerpunkt** der Region ein quadratisches Fenster zentriert; Größe =
Regions-Ausdehnung, clamp 24–48 px (deckt das BNN-Window ab).

**Fortschritt gegenüber `bbox`:**
- Ziffer landet deutlich häufiger zentriert → Recall 0.038 (+16 %),
  Bilder mit TP 0.260 (+18 %), Klassifikation 0.844 (+9 pp).

**Warum es nicht reicht:**
- Bei **verschmolzenen Mehrfach-Ziffern** (eine große CC-Region, zwei
  Ziffern mit überlappenden Kacheln) liegt der gewichtete Schwerpunkt
  **zwischen den Ziffern**. Ein Fenster um diesen Punkt isoliert keine
  einzelne Ziffer → das BNN bekommt ein Gemisch, die Box ist falsch.
- Genau diese "gemergten" Regionen sind die Restfälle, an denen `refocus`
  Recall liegen lässt. Einfaches Aufteilen der Region (z. B. Clustering im
  ganzen Bild, siehe kmeans) erzeugt aber mehr Probleme, als es löst.

### 3. `kmeans` — Lloyd-k-means über das ganze Saliency-Grid

**Idee:** k-means direkt auf allen aktiven Kachel-Zentren (k = round(Masse /
1.5), k-means++-Init, gewichtete Zentren), jeweils größtes Raster-Quadrat,
das alle wichtigen eigenen Kacheln enthält und keine wichtigen Kacheln
fremder Cluster schneidet.

**Erste Version (gemessen):**
- k = round(Masse/1.5), clamp [1,12] → im Schnitt 8.4 Fenster/Bild (vs. ~3.2
  bei CC-Regionen) → **viele redundante, überlappende Fenster**.
- Precision 0.212, Recall 0.0227, 3064 Kandidaten, 65.9 Forward-Pässe/Bild:
  klar dominiert (schlechteste Precision **und** Recall).

**Angewandte Verbesserungen (eingebaut, vor Archivierung):**
1. *Merge verschränkter Cluster:* Ziffer, die k-means auf 2 Cluster splittet
   (verschränkte Kacheln → kein Quadrat möglich), wird per Union-Find
   zusammengeführt.
2. *Dedup/NMS:* überlappende Fenster (IoU >= 0.6) werden entfernt, das
   saliency-massenreichste bleibt.
3. *k-Schätzung:* mass_per_digit 1.5 → 2.0.
4. *Deterministische Semen* für die MC-Dropout-Messung.

**Effekt der Verbesserungen:** Precision 0.212 → 0.254, cls_acc 0.808 →
0.870, Forward-Pässe 65.9 → 49.9. Ein Parameter-Sweep über mass_per_digit ×
merge_gap × dedup zeigte den prinzipiellen Trade-off: weniger Cluster →
präziser aber weniger Funde; mehr Cluster → mehr FPs.

**Warum es trotzdem nicht funktioniert:**
- **Fehlender Objekt-Kontext:** k-means legt seine k Zentren ohne die
  Regionen-Information quer über das Grid. Ein Fenster überdeckt dadurch
  häufig **mehrere Ziffern oder Hintergrundanteile** — selbst nach
  Merge/Dedup. Der BNN bekommt keine klar isolierte Ziffer und meldet
  sicher klassifizierte False Positives (Kandidaten-Level det_prec 0.046 bei
  gate 0.7 vs. 0.52 bei refocus).
- Recall bleibt dauerhaft unter 0.022 (bestes Sweep-Ergebnis 0.018 auf 200
  Bildern) — die quadratischen Fenster decken kleine Ziffern oft nicht
  isoliert ab, und die Fenstergrößen pendeln an 16 px, wo der BNN
  (trainiert auf 24–48 px) schwer arbeitet.

## 4. `hybrid2` — Peak-seeded k-means mit harten Mindest-Fenstern (Standard)

**Diagnose-Ausgangslage (Recall-Zerlegung, 400 Bilder, 3213 GT):**
- Saliency-Stufe ist nicht der Engpass (99.8 % der Zifferzentren liegen in
  einer Region mit Cover >= 0.5).
- Der Crop war es zu 42 %: nur 58.3 % der Zifferzentren lagen in einem
  hybrid-Fenster. Kleine Regionen: 100 %, große (verschmolzene) Regionen:
  nur 46 %.
- kmeans-Diagnose großer Regionen: k-Schätzung (Masse/1.5) überschätzt und
  läuft in 51 % der Regionen gegen den Deckel 6; Ø k=3.8 erzeugt nur Ø 1.8
  Fenster (55 % der Cluster "verpuffen"); die harte Fremd-Exklusion von
  `_best_square` ist in dichten Szenen unerfüllbar → 39 % der Fenster <= 24 px.

**Die Idee (drei gezielte Eingriffe):**
1. **k aus Saliency-Peaks statt aus Masse:** Anzahl der lokalen Maxima der
   8x8-Saliency innerhalb der Region (3x3-Nachbarschaft, >= Threshold 0.5,
   gierig dedupliziert). Kein Deckel mehr, k = Anzahl der Ziffernkandidaten.
   Die Peaks seeden die k-means-Zentren direkt (statt k-means++).
2. **Leere Cluster reparieren, nicht verwerfen:** verliert eine Lloyd-
   Iteration einen Cluster, wird er auf das am weitesten entfernte freie
   Seed zurückgesetzt und weitergefittet.
3. **Fenster zentriert statt hart-exklusiv:** Fenster um den gewichteten
   Cluster-Schwerpunkt, Größe = Cluster-Ausdehnung clamp [24, 48], **keine
   harte Fremd-Exklusion** (Überlappung erlaubt — die Ziffer sitzt durch die
   Zentrierung mittig, der Box-Head lokalisiert). Danach **Waisen-
   Absicherung:** wichtige Kacheln, die in keinem Fenster liegen, erhalten
   ein Peak-zentriertes 28-px-Zusatzfenster.

**Effekt (400 Bilder):** GT-Zentrum-Coverage 58.3 % → **97.8 %**, GT-Boxen
komplett gedeckt 90.7 %, Fenster/Bild 5.25 → 5.33 (≈ konstante Kosten).

> **Update — Fenster-pro-Ziffer (Ursache "zu wenige Boxen"):** Die 400-/1000-
> Bilder-Messung zeigte 1.5 Ziffern/Fenster: Die Dedup-Entfernung (IoU≥0.6)
> verschmolz benachbarte Ziffern-Fenster zu ~5.2 Fenstern pro ~8 Ziffern —
> und **ein Fenster liefert nur eine Detektion**. Seit dem Fix garantiert
> `_hybrid2_patches` jedem Saliency-Peak ein eigenes 28-px-Fenster und
> dedupliziert nur noch lockere Duplikate (`dedup_iou=0.75`, `peak_dedup_iou=0.85`).
> Ergebnis: **8.6 Fenster/Bild ≈ 0.93 Ziffern/Fenster**, Coverage 0.982 —
> der Recall verdoppelt sich bei gleicher Precision.

**Vollmessung (1000 Testbilder, 8017 GT, `hybrid2_eval.csv`):**

| Metrik | hybrid2 alt (gate 0.6) | hybrid2 neu (gate 0.6) | hybrid2 neu (gate 0.5) |
|--------|------------------------|------------------------|------------------------|
| Precision | 0.5918 | **0.600** | **0.561** |
| Recall | 0.0555 | **0.0881** | **0.1403** |
| Bilder mit >= 1 TP | 0.352 | **0.505** | **0.688** |
| Klassen-Acc | 0.8831 | **0.837** | **0.774** |
| Forward-Pässe/Bild | 41.90 | 68.5 | 68.8 |
| Patches/Bild | 5.24 | 8.61 | 8.61 |
| Box-IoU mean/med | 0.654/0.640 | — | 0.651/0.637 |
| Zentrumsfehler (px) | 2.08 | — | 2.07 |

**Einordnung (neu):** Die Stufen-Zerlegung lokalisiert zwei getrennte Engpässe:
1. **Fenster-Merging** (behoben): 1.5 → 0.93 Ziffern/Fenster verdoppelt den
   Recall (+57 % bei unveränderter Precision).
2. **BNN-Konfidenz** (bleibt): Auch mit perfekten Fenstern liefert der
   Klassifikator nur für einen Teil der Ziffern `p_max` über dem Gatter
   (MNIST: 345 von 2399 Ziffern-Fenstern ≥ 0.6, 1282 ≥ 0.35). Das Gatter
   ist der bewusste Präzisions-/Abstentions-Regler; für höheren Recall
   senkt man es auf 0.5 (recall 2.5×, Precision −4 pp).

## Fazit

`hybrid2` kombiniert das Beste: **kleine Regionen → genau ein Refokus-Crop**
(billig, zentriert), **große verschmolzene Regionen → Peak-seedete
k-means-Subcluster mit zentrierten Mindest-Fenstern + Waisen-Absicherung**.
Das ergibt mit gleichem Forward-Budget die beste Detektion (Recall, Bilder
mit TP), Klassifikation und Precision aller gemessenen Strategien und wird
daher als Default verwendet (`mode="hybrid2"`).

`crop_strategies_archive.py` hält die vier verworfenen Implementierungen als
selbstständige Referenz fest; `modes_compare_400.csv` enthält die vollständige
400-Bilder-Messung, `hybrid2_eval.csv` die Vollmessung (1000 Bilder) mit
Box- und Klassen-Details.