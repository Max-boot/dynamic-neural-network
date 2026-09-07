# Auswertung der TinyML-Artikelserie von Thommaskevin

Quelle: `https://medium.com/@thommaskevin` (TinyML-Artikelserie, Repo `thommaskevin.github.io/TinyML/`)

**Zweck dieser Datei:** Konsolidierte, unabhängige Bewertung aller Artikel der Serie im
Hinblick auf das Ziel des Dynamic-Neural-Network-Projekts — **präzise/akkurate Netze auf
dem ESP32-CAM** (Bildverarbeitung). Die Zusammenfassung wurde als Markdown ins Repo
übernommen, damit sie nicht nur im Chat, sondern dauerhaft versioniert vorliegt.

---

## Wichtigste Erkenntnis (Querschnitt)

**Keiner der Artikel behandelt direkt CNN-Bildverarbeitung auf der CAM.** Alle Beispiele
sind 1D/2D-Sensordaten (Regression, Klassifikation kleiner Feature-Vektoren). Für *echte*
Bilder ist nur ein **Feature-Extraktor davor** (HOG, Histogramme, ein vorgeschaltetes CNN)
+ nachgelagerter Leichtklassifikator sinnvoll. Das ist eine gemeinsame, wichtige
Einschränkung fast aller Artikel.

---

## Relevanz für das Ziel — im Detail

### 🟢 Hoch / Direkt brauchbar

**1. Adaptive Neuro-Fuzzy (ANFIS)** — *Bester Kandidat für kleine, präzise Klassifikation*
- Erste Architektur mit **sehr wenigen Parametern** (25 Regeln ≈ 106 Parameter, wenige KB
  Flash), no framework, reiner C++ — ideal für ESP32.
- **Vorteil für das CAM-Ziel:** Werden vorher Merkmale aus dem Bild extrahiert (z. B.
  Farb-/Kanten-Histogramme), kann ANFIS präzise und interpretierbare Entscheidungsgrenzen
  lernen.
- **Grenze:** Regelzahl explodiert exponentiell mit Inputs (2×5=25, 4×5=625 Regeln). Bei
  Pixeldaten als Input nicht sinnvoll — nur auf wenige extrahierte Features.

**2. Causal Decision Trees** — *Bestätigt die ROI-Idee (Pre-Filter)*
- Der Deployment-Pfad auf ESP32 funktioniert nachgewiesen (`generate_ino(board='esp32')`).
  Inferenz = nur O(Tiefe) Vergleichs-Operationen, also nahezu kostenlos.
- **Direkt relevant zur "Decision Tree grenzt Bereiche ein, dann teures NN"-Idee:** ein Baum
  passend auf "interessante Region ja/nein" wäre ein nahezu kostenloser Gating-Vorlauf.
- **Achtung:** Der Artikel behandelt *kausale* Effekte (Treatment, CATE), **kein**
  Bild-ROI-Gating und keine Präzision/Recall-Zahlen. Der Baum müsste als binärer
  "Region relevant?"-Klassifikator umgewidmet werden. Technisch machbar, aber ohne Zahlen
  aus dem Artikel.

### 🟡 Mittel / Nur für Sensoren, nicht für Bilder

**3. Bayesian NN** — *Für Confidenz/Fehlklassifikations-Schutz* (teils brauchbar)
- Liefert Unsicherheit (epistemisch/aleatorisch) + OOD-Erkennung ("Vorhersage verwerfen,
  wenn σ² > τ"). Das *wäre* nützlich, um auf der CAM falsche Detektionen zu vermeiden.
- **Aber:** 2× Parameter (μ+ρ), braucht S Monte-Carlo-Passes (höhere Latenz), nur MLP gezeigt
  (kein Conv). Für Bilder müssten Bayessche Conv-Layer selbst gebaut werden. Quantisierung
  fehlt.

**4. Neural Additive Models** — *Nur als schlanker Feature-Klassifikator*
- Winziges natives C, interpretierbar. Aber nur tabellarische Skalare; **nicht** für rohe
  Pixel. Passend als zweite Stufe nach Feature-Extraktion.

**5. Spiking NN** — *Energiesparend, aber für CAM schwer*
- Event-driven = sehr sparsam, aber nur `SpikingLinear` (keine Conv-Layer), Rate-Encoding
  verliert Auflösung (T=25 ≈ 4 bit), Latenz = T× sequenziell, kein INT8. Für präzise
  Bildklassifikation auf CAM nicht der richtige Weg.

### 🔴 Gering / Nicht relevant für CAM

**6. Liquid NN** — nur Zeitreihen, keine räumliche Verarbeitung.
**7. Graph Conv NN** — nur Graph-Daten (Adjazenzmatrix). Für Bilder bräuchte es erst einen
künstlichen Region-Graph — Overkill.
**8. RNN** — nur 1D-Sequenzen; höchstens Video-Sequenzanalyse.
**9. Quantile Regression** — nur Unsicherheits-/Intervall-Schätzung bei Regression.
**10. GAN** — generativ, nicht für Klassifikation; nur für Augmentation/Anomaly-Detection.

---

## Fazit für das Projekt

Aus allen 10 Artikeln lassen sich **drei verwertbare Bausteine** ableiten:

1. **Die Decision-Tree-ROI-Idee wird von den Artikeln 1+2 gestützt:** Eine billige
   Vorsperre (Decision Tree / Fuzzy) vor dem teuren NN. Auch der Causal-Tree-Deployment-Pfad
   funktioniert auf ESP32.
2. **Feature-Extraktion + Leichtklassifikator** (ANFIS/NAM) ist der einzige Weg, diese
   Konzepte für echte Bilder nutzbar zu machen — nicht rohe Pixel durch ein
   Tiny-Fuzzy/NAM-Modell.
3. **Für tatsächlich akkurate Bild-Objekterkennung auf der CAM bleibt FOMO/MobileNetV2 + INT8
   der bewährte Weg** (wie zuvor recherchiert) — die Artikel von Thommaskevin liefern dazu
   selbst nichts Besseres, bestätigen aber die Toolchain (`generate_ino`) für die
   Leichtmodelle.

### Abgleich mit der bislang erprobten Projekt-Erkenntnis

Die Serie deckt sich mit der zentralen empirischen Erkenntnis dieser Session: **Einfache
Methoden schlagen komplexere beim MCU-Deployment.** Auch hier — im Vergleich der
Wichtigkeits-Metriken (siehe `Report/Benchmarks/README_BENCHMARKS.md`) — schlug die einfache
L1-Norm den trainierten Meta-Scorer, und ein RL-Pruner blieb hinter der Heuristik zurück.
Für präzise Netze auf stark beschränkten MCUs sind deshalb **strukturiertes Pruning nach
L1-Magnitude + physisches Kompaktieren + INT8-Quantisierung** der robuste, industriell
bewährte Weg — mehrstufige "Saliency → Decision Tree → NN nur auf Regionen"-Pipelines
lohnen nur auf Geräten mit mehr RAM oder als Forschung.

---

## Verwandt

- Projekt-README: `/Code` (Repo-Root `README.md`, `PROJECT_REPORT.md`).
- Benchmark-/Vergleichsdokumentation: `Report/Benchmarks/README_BENCHMARKS.md`.
- Recherche-Quelle: `https://medium.com/@thommaskevin` und `thommaskevin.github.io/TinyML/`.
