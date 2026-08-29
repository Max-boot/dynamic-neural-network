# Dynamic Neural Network — Current State Report & Roadmap

**Date:** 28 August 2026
**Basis:** Measured outputs from `EXPERIMENT_LOG.md`, `hybrid_output.txt`, `deployment_metadata.json`, `deploy_esp32/README.md`, and source inspection. All numbers below are taken from recorded runs, not estimates.

---

## 1. What the project is

A **biologically-inspired self-organizing neural network** for **human detection / person-position classification**, targeted at deployment on a memory-constrained **ESP32-CAM microcontroller**. The headline research angle is replacing classic backpropagation with biological learning rules: STDP, homeostatic plasticity, structural growth/pruning rules ("neurons that fire together, wire together"), and developmental stages.

Concretely, it is a **7-class image classifier** on the YOLO-HiVis dataset:
- `MULTI` (multiple persons), `LEFT`, `RIGHT`, `SMALL` (too far), `BIG` (too close), `CENTER`, `NONE` (no person) → maps to a robot/drone action ("TURN LEFT", "MOVE CLOSER", "SEARCH", etc.).

The project has evolved through **14+ documented experiment versions** (v1→v10.x) plus a family of "hybrid" CNN+Hebbian and TinyYOLO variants.

---

## 2. Measured capabilities

### 2.1 Dataset reality (measured)
- **YOLO-HiVis**: 7,937 images, 96×96 grayscale, YOLO-format labels (verified: 7,937 images and 7,937 label files on disk).
- Class distribution (train split, from run logs):

| Class | Train | Val | Notes |
|-------|------:|----:|-------|
| MULTI | 2,465 | 425 | 3rd most common |
| LEFT | 105 | 22 | **severe minority** |
| RIGHT | 121 | 24 | **severe minority** |
| SMALL | 64 | 14 | **rarest** |
| BIG | 1,592 | 248 | common |
| CENTER | 277 | 47 | minority |
| NONE | 2,123 | 410 | 2nd most common |

- Other datasets present: `human detection dataset` (0: 362, 1: 559 images — binary) and `COCO` (1,500 images), but all *current* measured results are on YOLO-HiVis.

### 2.2 Best measured model (deployed)
From `deployment_metadata.json` (the artifact actually exported to ESP32):
- **Architecture:** PCA-32 (dimensionality reduction) → MLP 32→16→7
- **Params:** 647 synapses
- **Best validation accuracy:** **51.1%** (0.5107)
- **PCA variance retained:** 70.2%
- **Flash footprint:** ~300 KB (PCA mean 9 KB + components 295 KB + MLP 2.6 KB)
- **Inference latency (240 MHz):** ~1.2 ms — exceeds the <2 ms goal.

### 2.3 Per-class capability gap (measured — the key weakness)
The overall 51% hides extreme class imbalance in *usefulness*:

| Class | Recall (best model) |
|-------|--------------------:|
| MULTI | 80.7% |
| BIG | 23.7% |
| NONE | 49.2% |
| RIGHT | 8.5% |
| CENTER | 3.2% |
| LEFT | 2.3% |
| SMALL | **0.0%** |

**Interpretation:** The model is essentially a *detector* (is there a person / multiple people / big person), not a *position/localization* model. The physically meaningful classes for navigation (LEFT, RIGHT, CENTER, SMALL) are near-random. This is the single most important limitation.

### 2.4 Other measured architecture results
| Model | Params | Val acc | Notes |
|-------|-------:|--------:|-------|
| PCA-32 + MLP (v10.10-14) | 647 | **51.1%** | best deployed |
| Patch net 7×7 (v10.5) | 2,800 | 47.9% | | 
| CNN+Hebbian backprop head | ~7K | 45.5% | from `hybrid_output.txt` |
| TinyYOLO @96 | 43.3K | 41.6–41.7% | from `hybrid_output.txt` |
| LDA / PCA+LDA (v10.12) | 91 | 34.3 / 48.8% | |
| Ensemble PCA+Patch (v10.13) | 3,447 | 50.0% | |
| **CNN+Hebbian forward (Hebbian-only)** | — | **1.2%** | **broken / nonfunctional** |

---

## 3. Verified limitations (measured, not guessed)

1. **The core research claim (Hebbian/biological learning) does not actually work.**
   The `hybrid_output.txt` run measured the "Hebbian forward" path at **1.2% accuracy** — *below random chance* (12.5% would be... no, 1/7 ≈ 14%, so 1.2% is catastrophic). This is the headline claim of the project ("self-organizing, STDP, no backprop") and it is nonfunctional. Note: the README's v5 architecture (160×120 RGB, 57,600 input neurons, GPU-STDP) describes an *aspirational* design; **no persisted measurement exists for it.**

2. **Minority classes are unsolvable with the current data + approach.**
   SMALL (86 samples), LEFT (394), RIGHT (418) across the *entire* historical record — every method fails on them. This was explicitly concluded in the experiment log ("evidence: data-limited"). No amount of architecture tweaking fixes a data problem.

3. **Class imbalance is structural.** NONE (32%) + MULTI (36%) = ~68% of the data. A model can hit high "accuracy" by mostly correct guessing these, which is why overall % looks better than the model actually is.

4. **No version control.** The folder is not a git repository. 52 top-level files, most of them one-off experiment scripts (`hybrid_v3..v9`, `tiny_yolo_*`, `mnist_*`, `main.py`) with no clear "current best" entry point. The README describes versions (v1–v5) that do not match what's actually on disk (the real lineage is v10.x + hybrid variants).

5. **Broken/dead code and unpersisted results.**
   - `hybrid_cnn_hebbian.py` line 615 has a `NameError: name 'cn' is not defined` (visible in `hybrid_output2.txt`) — a crash in the main run script.
   - The best-looking RGB results (v8 @192×192 claiming best_val **0.563**, and 256×256 variants) were **never persisted to any log/metadata** — only visualization PNGs exist (`visualizations_v8/`, `visualizations_yolo256/`). Their numbers are unverifiable and unused.
   - `export_log.txt` is a binary/garbled file.
   - `main.py` (the MNIST v1 backprop demo) and the huge `v9_human_tracking_multihead.py` (1,300+ lines on GPU+CPU) look like legacy research, with 128×128 inputs incompatible with what's deployed.

6. **Reproducibility risk.** No git history, no saved training run config (seed is hard-coded SEED=42 everywhere but scripts differ). The same class counts appear in many logs because each script re-derives labels with `[1,5,2]` conventions — small inconsistencies across files make it hard to trust cross-file comparisons.

7. **Flash-spending is lopsided.** The deployed model spends ~295 KB of ~300 KB on the *PCA components matrix* (a fixed linear projection) and only ~2.6 KB on the actual learned classifier. That's efficient for shipping but means the "model" is mostly off-device-derived preprocessing.

---

## 4. Evaluation — what's good

- **Honest, disciplined experiment log.** `EXPERIMENT_LOG.md` records what was tried *and what failed* with real numbers (44.6% → 51.1%). This scientific record-keeping is genuinely good and the main reason the project is assessable at all.
- **Real deployment achieved and measured.** A model was exported to ESP32 (int8 quantization, ~300 KB flash, ~1.2 ms inference), with C headers, a PlatformIO project, and a camera inference sketch. That is a concrete, verified deliverable.
- **Clear problem framing.** Multi-head outputs (position/size/confidence/multiple) is a smart decomposition for the navigation use-case, and the priority hierarchy (MULTI > LEFT/RIGHT > SMALL/LARGE) is sensible.
- **Parameter efficiency.** 647 synapses for 7 classes with a PCI footprint is genuinely small and the PCA-dimension sweep (v10.11) confirmed the right scaling insight (32 > 64/96/128).
- **Genuine domain breadth.** The project legitimately touches neuroscience, embedded systems, and ML — a strong school showcase (as the `email_vorlage.md` shows).

## 5. Evaluation — what needs improvement (biggest impact first)

1. **Fix or drop the Hebbian-only claim.** 1.2% is a showstopper for the stated research thesis. Either (a) make the biological path competitive, or (b) re-scope the thesis to "biological-structure *as a compact regularizer* on top of a working supervised head" (the CNN+Hebbian 45.5% backprop head is the closest to working). Currently the marketing (README v5) overpromises relative to measured reality.

2. **Solve the data problem before the architecture problem.** The experiment log already proved LEFT/RIGHT/SMALL are data-limited. Recommended: (a) more data for those three classes, (b) horizontal-flip augmentation *with label swap* is the cheapest high-value fix (flips LEFT↔RIGHT, SMALL↔BIG), (c) or reframe to fewer classes where the data actually supports it.

3. **Persist and unify results.** Add a single `results.json`/`results.md` artifact updated by every run, and surface it (may already exist as `deployment_metadata.json` — extend it). Without it, the promising v8/v9/256 numbers are untraceable.

4. **Introduce version control + a clean entry point.** `git init`, prune dead scripts, and make one canonical `train.py` + `export_esp32.py` with a CLI. Right now an evaluator cannot tell which file is "the" model.

5. **Fix the crashing script** (`hybrid_cnn_hebbian.py:615`).

6. **Add a validation/consistency harness** — one shared `labels.py` used by all scripts so per-class counts and label conventions can't drift between files.

---

## 6. Roadmap to success

"Success" needs defining. Two credible definitions:
- **A (research thesis):** the biological network genuinely learns (beats/approaches the supervised baseline).
- **B (engineering product):** a correct, reliable, tiny on-device person-position classifier.

These need different roadmaps. Recommended path below hedges toward B (reachable with real data you can measure), keeping A as a stretch research thread.

### Phase 1 — Consolidation (1–2 weeks)
- `git init`; commit current state; move one-off scripts to `experiments/`.
- Add a single shared `labels.py` + dataset loader; delete/archive `main.py`, `mnist_*`.
- Fix `hybrid_cnn_hebbian.py:615`.
- Build one `train.py` that logs **JSON** (params, per-class recall, val acc, seed) to `results/`.
- **Gate:** re-run PCA-32 and reproduce **51% ± 1** — establishes a trustworthy baseline.

### Phase 2 — Data (2–4 weeks) — highest ROI
- Collect/augment LEFT, RIGHT, SMALL (target ≥400–500 each, ~10× current SMALL).
- Add HFlip-with-label-swap + small shift augmentation.
- **Gate:** re-measure minority recalls; aim to lift SMALL 0%→>20%, LEFT/RIGHT >40%.

### Phase 3 — Architecture (2–4 weeks) — pick ONE direction
- **Direction B (recommended):** keep the PCA/CNN front-end + multi-head, but introduce explicit **spatial/positional features** so LEFT/RIGHT/CENTER are learnable from the 70%-variance PCA space or from a tiny conv. Evaluate the unpersisted v8/v9/256 architectures properly and pick the best by measured val acc **under the same seed/split**.
- **Direction A (research):** repair the Hebbian path in isolation — first get it above chance on a *balanced 2-class* proxy (person present), then 3-class, then the full 7-class. Set the expectation that bio-alone may cap below the supervised head; publish it as such rather than as a working replacement.
- **Gate:** one model reproducibly ≥55% overall AND ≥40% on each non-NONE/MULTI class on the fixed split.

### Phase 4 — Deployment + validation (2–3 weeks)
- Fold BatchNorm into convs for int8; reclaim the 295 KB PCA spend by quantizing components to int8 (already done) or shrinking to 16–24 components — re-measure accuracy vs flash.
- Add an on-device confidence/reliability test on a held-out real camera set (not just the training distribution).
- Document latency, RAM, power, and per-class failure modes in a single `DEPLOY.md`.
- **Gate:** live ESP32 demo: correct LEFT/RIGHT/SMALL/BIG action in front of camera.

### Phase 5 — Science write-up & showcase (1–2 weeks)
- Produce a short report: measured comparison table (all five architectures, same split), a clear statement of what the biological mechanisms *can* and *cannot* do given the data, and the reproducibility recipe.
- Update README to reflect reality (remove unverified v5 claims or mark as "proposed").

### Success criteria (measurable, all-or-nothing)
1. Reproducible baseline: PCA-32 ≥ 51% on the fixed split. ✅/❌
2. Minority lift: SMALL > 20%, LEFT & RIGHT > 40%, CENTER > 25% recall.
3. Overall ≥ 55% on an *unseen test split* (today: ~51% measured).
4. One ESP32 artifact with quantified on-device latency ≤ 2 ms and per-class behavior on live camera images.
5. A single versioned `results/` archive + git history showing every claim is backed by a recorded run.

---

## 7. Bottom line

The project is **scientifically well-kept and engineering-realistic but currently under-delivers on its headline claim.** What works and is measured: a compact, deployed, fast on-device person *detector* (~51%, running on real ESP32 hardware). What does not yet exist as a working, measured artifact: the *self-organizing biological network* the project is about (1.2% measured), and a solution to the *minority-position classes* that matter for actual navigation.

The shortest honest path to "success" is: **fix the data, then the architecture, then ship and document one verified artifact** — while being candid that the biological-learning thesis remains unproven and needs its own focused campaign.
