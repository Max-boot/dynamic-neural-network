"""
ML entscheidet selbst: adaptive Schichtgroessen via Synapsen-Budget
===================================================================
Unterschied zu den bisherigen Benchmarks:
  Frueher wurde ein FESTES Ziel vorgegeben (z.B. [96,48,24]).
  Hier gibt das ML-System nur ein SYNAPSEN-BUDGET vor (SYN_TARGET, einstellbar;
  z.B. 81.264 ~ -66.5% oder 40.000 ~ -83.5%) und entscheidet selbst, WELCHE
  Neuronen und in WELCHER Schicht geprunt werden, bis das Budget erreicht ist.

Kernidee:
  - Nicht alle Neuronen sind gleich "teuer":
      * Neuron in Schicht 1  -> entfernt eingehende 784 + ausgehende (n) Synapsen
      * Neuron in Schicht 3  -> entfernt nur eingehende (n) + ausgehende (24)
    Ein Neuron der ersten Schicht spart also VIEL mehr Synapsen.
  - Der Meta-Scorer belegt jedes Neuron mit einer Wichtigkeit (0..1 je Schicht,
    layout-invariant; hoher Wert = wichtig).
  - Greedy-Budget-Pruning: Es wird ITERATIV das Neuron mit der besten
    "Wichtigkeits-Einsparung" entfernt:
        prio(j) = (1 - score_j) / syn_ersparnis(j)
    (unwichtigstes Neuron pro gesparter Synapse zuerst). Stopp, sobald das
    Synapsen-Budget erfuellt ist.

Ergebnis / Beobachtung:
  Die dabei entstehenden SCHICHTGROESSEN sind das "Verhalten" des ML:
  erkennt es z.B., dass spaete Schichten relativ unwichtig und guenstiger zu
  prunen sind (niedrigere Ersparnis), und reduziert diese staerker?

Ausgabe:
  - Konsolen-Log (Schichtgroessen vor/nach, Synapsen)
  - RESULT_ml_budget_pruning.csv
  - RESULT_bild_ml_budget_pruning.png
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn.functional as F
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
MT_PATH = r"D:\Dynamic Neural Networt (DNN)\Code\structural_ml_metatrain.py"

def _load_mt():
    spec = importlib.util.spec_from_file_location("mt", MT_PATH)
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt

mt = _load_mt()
device = mt.device
train_loader = mt.train_loader
test_loader = mt.test_loader
print(f"Device: {device}", flush=True)

LAYOUT = [256, 128, 64]
SYN_TARGET = 40000          # Ziel-Synapsen (BUDGET, ~ -83.5% von 242304)
SEEDS = [42, 2024]
WARM = 6                     # Epochen vor dem Pruning (voll trainiert)
FINETUNE = 6                 # Epochen nach dem Kompaktieren


def load_meta_scorer():
    candidates = ["models/neuron_scorer_meta.pt",
                  os.path.join(_HERE, "models", "neuron_scorer_meta.pt"),
                  r"D:\Dynamic Neural Networt (DNN)\Code\models\neuron_scorer_meta.pt"]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    scorer = mt.NeuronScorer(n_feat=6).to(device)
    scorer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    scorer.eval()
    return scorer


def collect_scores(model, scorer):
    """Bewertet alle aktiven Neuronen mit dem Meta-Scorer und liefert
    (scores_global, korr). scores_global: np-Array der Laenge tot_hidden."""
    model.reset_activity_buffer()
    model.reset_class_stats()
    for xb, yb in list(train_loader)[:3]:
        xb, yb = xb.to(device), yb.to(device)
        model(xb)
        model.update_class_stats(xb, yb)
    corr = model.compute_correlation_matrix()
    feats = mt.extract_features(model, corr)          # (tot_hidden, 6)
    with torch.no_grad():
        scores = scorer(torch.from_numpy(feats).float().to(device)).cpu().numpy()
    return scores, corr


def syn_savings(model, layer_idx, neuron_local):
    """Ersparnis an Synapsen, wenn dieses NeurON entfernt wuerde."""
    counts = model.neuron_counts          # aktuelle aktive Neuronen je Schicht
    L = layer_idx
    sz = model.HIDDEN_SIZES[L]
    in_dim = 28 * 28 if L == 0 else counts[L - 1]
    out_dim = model.n_classes if L == model.n_layers - 1 else counts[L + 1]
    return in_dim + out_dim


def budget_prune(model, scores, syn_target, floor_frac=0.15, floor_min=8):
    """
    Greedy-Budget-Pruning ueber alle Schichten nach Wichtigkeit+Ersparnis,
    bis active_synapses <= syn_target.

    Bedeutung der Regel (eigentliche ML-Entscheidung):
      - Der Meta-Scorer liefert je Neuron seine Wichtigkeit (0..1).
      - Ein Neuron der ersten Schicht entfernt VIEL mehr Synapsen als eines
        der letzten Schicht (Ersparnis = eingehende + ausgehende Synapsen).
      - Regel: prune das Neuron mit prio = (1 - score) * Ersparnis zuerst.
        Das ML entfernt also bevorzugt TEURE (fruehe) UNWICHTIGE Neuronen,
        weil das das Budget am effizientesten erfuellt.
      - Ein FLOOR pro Schicht (floor_frac der Originalgroesse, mind.
        floor_min) verhindert, dass eine Schicht komplett kollabiert, damit
        ein ausfuehrbares Netz erhalten bleibt (siehe auch die Diskussion).

    Modifiziert die neuron_masks des Modells.
    Liefert (finale keep_counts, cost_history).
    """
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(floor_frac * o), floor_min) for o in orig]
    starts = model.hidden_start_indices
    cost_log = []

    while model.active_synapses > syn_target:
        counts = model.neuron_counts
        # Falls bereits alle Schichten auf dem Floor sind -> stoppen
        if all(c <= f for c, f in zip(counts, floor)):
            break
        cand_layer = []
        cand_local = []
        cand_score = []
        cand_save = []
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue          # Schicht nicht weiter schrumpfen
            mask = model.neuron_masks[L].cpu().numpy()
            active_idx = np.where(mask > 0)[0]
            lo = starts[L]
            for nl in active_idx:
                cand_layer.append(L)
                cand_local.append(nl)
                cand_score.append(scores[lo + nl])
                in_dim = 28 * 28 if L == 0 else counts[L - 1]
                out_dim = model.n_classes if L == model.n_layers - 1 \
                    else counts[L + 1]
                cand_save.append(in_dim + out_dim)
        if not cand_layer:
            break
        cand_score = np.array(cand_score)
        cand_save = np.array(cand_save)
        # Prioritaet: unwichtiges UND teures Neuron zuerst
        prio = (1.0 - cand_score) * cand_save
        best = int(np.argmax(prio))
        L, nl = cand_layer[best], cand_local[best]
        mask = model.neuron_masks[L].clone()
        mask[nl] = 0.0
        setattr(model, f"neuron_mask_{L}", mask)
        cost_log.append(model.active_synapses)
        new_counts = model.neuron_counts
        if len(cost_log) % 25 == 0 or model.active_synapses <= syn_target:
            print(f"    budget-step: syn={model.active_synapses:,}  "
                  f"layers={new_counts} Scr={scores[starts[L]+nl]:.2f} "
                  f"save={cand_save[best]:,}",
                  flush=True)

    return model.neuron_counts, cost_log


def train_epochs(model, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def compact_dynamicmlp(src, keep_counts, keep_idx_lists):
    """Erstellt ein NEUES physisch kompaktiertes DynamicMLP."""
    nc = mt.DynamicMLP(keep_counts).to(device)
    src_l = src.layers
    dst_l = nc.layers
    with torch.no_grad():
        for i in range(len(keep_counts)):
            ki = keep_idx_lists[i].to(device)
            if i == 0:
                dst_l[0].weight.copy_(src_l[0].weight[ki])
                dst_l[0].bias.copy_(src_l[0].bias[ki])
            else:
                k_prev = keep_idx_lists[i - 1].to(device)
                dst_l[i].weight.copy_(src_l[i].weight[ki][:, k_prev])
                dst_l[i].bias.copy_(src_l[i].bias[ki])
        # Ausgabeschicht
        k_last = keep_idx_lists[-1].to(device)
        dst_l[len(keep_counts)].weight.copy_(
            src_l[len(keep_counts)].weight[:, k_last])
        dst_l[len(keep_counts)].bias.copy_(src_l[len(keep_counts)].bias)
    return nc


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    scorer = load_meta_scorer()
    print(f"\n{'='*70}\n SEED {seed}\n{'='*70}", flush=True)

    model = mt.DynamicMLP(LAYOUT).to(device)
    train_epochs(model, WARM)
    acc_before = mt.measure_accuracy(model)
    syn_before = model.active_synapses
    print(f"  VOLL trainiert: acc={acc_before:.2f}% syn={syn_before:,} "
          f"layers={list(LAYOUT)}", flush=True)

    scores, corr = collect_scores(model, scorer)

    keep_counts, cost_log = budget_prune(model, scores, SYN_TARGET)
    syn_after_mask = model.active_synapses
    print(f"  Budget-Pruning fertig: syn={syn_after_mask:,} "
          f"layers={keep_counts}", flush=True)

    # Behaltene Indizes je Schicht aus den Masken extrahieren
    keep_idx_lists = [torch.nonzero(m > 0).squeeze(1).cpu()
                      for m in model.neuron_masks]

    # Physisch kompaktieren
    cmod = compact_dynamicmlp(model, keep_counts, keep_idx_lists)
    syn_compact = cmod.active_synapses
    # Fine-Tune
    train_epochs(cmod, FINETUNE)
    acc_final = mt.measure_accuracy(cmod)

    # Verlauf des Budgets
    full_cost = [242304] + cost_log

    print(f"  KOMPAKT: layers={keep_counts} syn={syn_compact:,} "
          f"acc={acc_final:.2f}%", flush=True)

    return {
        "seed": seed,
        "acc_before": acc_before,
        "acc_final": acc_final,
        "syn_before": syn_before,
        "syn_after": syn_after_mask,
        "syn_compact": syn_compact,
        "layers": keep_counts,
        "cost_log": full_cost,
    }


def main():
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS
    results = [run(s) for s in seeds]

    print("\n" + "=" * 72)
    print("  BEURTEILUNG: zu welchen Schichtgroessen entschied sich das ML?")
    print("=" * 72)
    for r in results:
        print(f"  Seed {r['seed']}:  {list(LAYOUT)} -> {r['layers']}   "
              f"syn {r['syn_before']:,} -> {r['syn_compact']:,}   "
              f"acc {r['acc_before']:.2f}% -> {r['acc_final']:.2f}%")

    # Durchschnittliche Schichtgroessen
    n_layers = len(LAYOUT)
    mean_layers = [np.mean([r["layers"][i] for r in results]) for i in range(n_layers)]
    print(f"  Mittelwert-Schichtgroessen: {[int(round(m)) for m in mean_layers]}")

    # Balkendiagramm: Schichtgroessen vorher/nachher (Mittelwert), plus Verlauf
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(n_layers)
    ax = axes[0]
    ax.bar(x - 0.2, LAYOUT, width=0.4, label="VOLL (vorher)", color="#888")
    ax.bar(x + 0.2, mean_layers, width=0.4, label="nach ML-Budget-Pruning",
           color="#d62728")
    for i in range(n_layers):
        ax.text(x[i] - 0.2, LAYOUT[i] + 2, str(LAYOUT[i]), ha="center", fontsize=8)
        ax.text(x[i] + 0.2, mean_layers[i] + 2,
                f"{mean_layers[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Schicht 1", "Schicht 2", "Schicht 3"])
    ax.set_ylabel("Neuronenzahl")
    ax.set_title("ML-Budget-Pruning: Schichtgroessen vor/nach (Mittelwert)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for r in results:
        ax.plot(range(len(r["cost_log"])), r["cost_log"], lw=1.5, alpha=0.6,
                label=f"seed {r['seed']}")
    ax.axhline(SYN_TARGET, color="#2ca02c", ls="--",
               label=f"Budget {SYN_TARGET:,}")
    ax.set_xlabel("Prune-Schritt")
    ax.set_ylabel("Synapsen")
    ax.set_title("Synapsen-Abbau bis zum Budget")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    png = os.path.join(_HERE, "RESULT_bild_ml_budget_pruning.png")
    fig.savefig(png, dpi=150)
    print(f"\n  Plot gespeichert: {png}")

    csvp = os.path.join(_HERE, "RESULT_ml_budget_pruning.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "acc_before", "acc_final", "syn_before",
                    "syn_after", "syn_compact",
                    "layer1", "layer2", "layer3"])
        for r in results:
            w.writerow([r["seed"], round(r["acc_before"], 3),
                        round(r["acc_final"], 3), r["syn_before"],
                        r["syn_after"], r["syn_compact"]] + r["layers"])
    print(f"  CSV gespeichert: {csvp}")


if __name__ == "__main__":
    main()
