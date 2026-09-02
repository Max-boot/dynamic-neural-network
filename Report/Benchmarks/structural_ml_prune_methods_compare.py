"""
VERGLEICH der Pruning-Methoden (gleiches Budget, gleiche Seeds)
================================================================
Drei Methoden werden unter IDENTISCHEN Bedingungen verglichen
(dieses Netz [256,128,64], Budget 40.000 Synapsen, gleicher Floor,
gleiche Seeds, gleiche Warm-/Fine-Tune-Epochen):

  A) ML-Score x Ersparnis        prio = (1 - score) * sav
     (alter Meta-Scorer + lineare Ersparnis-Wichtung)
  B) ML-Score x sqrt(Ersparnis)  prio = (1 - score) * sqrt(sav)
     (Ersparnis-Bias abgeschwaecht -> verteilt ueber Schichten)
  C) L1-Norm (TinyML-Standard)   prio = (1 - l1norm) * sav
     (Wichtigkeit = L1-Norm der eingehenden Gewichte pro Neuron;
      grosse Norm = wichtig)

Alle drei sind "greedy bis ans Synapsen-Budget, mit Floor pro Schicht".
Nur der Wichtigkeits-Score unterscheidet sich -> direkter, fairer
Vergleich der Wichtigkeits-Metriken.

Ausgabe:
  - Konsolenlog (Schichtgroessen + Accuracy je Methode und Seed)
  - RESULT_prune_methods_compare.csv
  - RESULT_bild_prune_methods.png
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
SYN_TARGET = 40000          # Budget (wie bisher)
SEEDS = [42, 2024]
WARM = 6                    # Epochen vor dem Pruning
FINETUNE = 6                # Epochen nach dem Kompaktieren


def load_meta_scorer():
    candidates = ["models/neuron_scorer_meta.pt",
                  os.path.join(_HERE, "models", "neuron_scorer_meta.pt"),
                  r"D:\Dynamic Neural Networt (DNN)\Code\models\neuron_scorer_meta.pt"]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    scorer = mt.NeuronScorer(n_feat=6).to(device)
    scorer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    scorer.eval()
    return scorer


def collect_ml_scores(model, scorer):
    """Meta-Scorer-Score je Neuron (0..1, hoeher = wichtiger)."""
    model.reset_activity_buffer()
    model.reset_class_stats()
    for xb, yb in list(train_loader)[:3]:
        xb, yb = xb.to(device), yb.to(device)
        model(xb)
        model.update_class_stats(xb, yb)
    corr = model.compute_correlation_matrix()
    feats = mt.extract_features(model, corr)
    with torch.no_grad():
        scores = scorer(torch.from_numpy(feats).float().to(device)).cpu().numpy()
    return scores


def collect_l1_scores(model):
    """L1-Norm der eingehenden Gewichte je Neuron, global auf [0,1] normiert.
    Hoeher = wichtigere (groessere) Norm."""
    norms = []
    for L in range(model.n_layers):
        w = model.layers[L].weight.detach().cpu()   # (out, in)
        norms.append(w.norm(p=1, dim=1).numpy())     # pro Neuron (out-Zeile)
    norm_full = np.concatenate(norms)
    lo = norm_full.min(); hi = norm_full.max()
    score = (norm_full - lo) / (hi - lo + 1e-8)      # 0..1, 1 = wichtigste
    return score


def greedy_prune(model, score, syn_target, use_sqrt,
                 floor_frac=0.15, floor_min=8):
    """Generisches greedy-Budget-Pruning mit einem Wichtigkeits-Score 0..1
    (1 = wichtig). Prio = (1 - score) * sav^pwr. Prunt bis zum Budget."""
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(floor_frac * o), floor_min) for o in orig]
    starts = model.hidden_start_indices

    while model.active_synapses > syn_target:
        counts = model.neuron_counts
        if all(c <= f for c, f in zip(counts, floor)):
            break
        cand_L, cand_nl, cand_prio = [], [], []
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue
            mask = model.neuron_masks[L].cpu().numpy()
            in_dim = 28 * 28 if L == 0 else counts[L - 1]
            out_dim = model.n_classes if L == model.n_layers - 1 else counts[L + 1]
            sav = in_dim + out_dim
            sav_term = np.sqrt(sav) if use_sqrt else sav
            for nl in np.where(mask > 0)[0]:
                cand_L.append(L)
                cand_nl.append(int(nl))
                cand_prio.append((1.0 - score[starts[L] + nl]) * sav_term)
        if not cand_L:
            break
        best = int(np.argmax(cand_prio))
        L, nl = cand_L[best], cand_nl[best]
        mask = model.neuron_masks[L].clone()
        mask[nl] = 0.0
        setattr(model, f"neuron_mask_{L}", mask)
    return model.neuron_counts


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
        k_last = keep_idx_lists[-1].to(device)
        dst_l[len(keep_counts)].weight.copy_(
            src_l[len(keep_counts)].weight[:, k_last])
        dst_l[len(keep_counts)].bias.copy_(src_l[len(keep_counts)].bias)
    return nc


def prune_and_measure(model, score, use_sqrt):
    """Prunt bis zum Budget, kompaktiert, feintunt und misst Accuracy.
    Liefert (keep_counts, syn_compact, acc)."""
    keep = greedy_prune(model, score, SYN_TARGET, use_sqrt)
    k_idx = [torch.nonzero(m > 0).squeeze(1).cpu() for m in model.neuron_masks]
    cmod = compact_dynamicmlp(model, keep, k_idx)
    syn = cmod.active_synapses
    train_epochs(cmod, FINETUNE)
    acc = mt.measure_accuracy(cmod)
    return keep, syn, acc


METHODS = [
    ("A_ml_x_sav",   True,  False, "ML-Score x Ersparnis"),
    ("B_ml_x_sqrt",  True,  True,  "ML-Score x sqrt(Ersparnis)"),
    ("C_l1_x_sav",   False, False, "L1-Norm x Ersparnis"),
]


def main():
    scorer = load_meta_scorer()
    rows = []

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        base = mt.DynamicMLP(LAYOUT).to(device)
        train_epochs(base, WARM)
        acc0 = mt.measure_accuracy(base)
        syn0 = base.active_synapses

        ml_score = collect_ml_scores(base, scorer)
        l1_score = collect_l1_scores(base)

        print(f"\n{'='*70}\n SEED {seed}  (voll: acc={acc0:.2f}% "
              f"syn={syn0:,})\n{'='*70}", flush=True)

        for key, use_ml, use_sqrt, lbl in METHODS:
            m = mt.DynamicMLP(LAYOUT).to(device)
            m.load_state_dict(base.state_dict())
            score = ml_score if use_ml else l1_score
            keep, syn, acc = prune_and_measure(m, score, use_sqrt)
            rows.append({
                "seed": seed, "method": key, "label": lbl,
                "acc0": acc0, "acc": acc, "syn0": syn0, "syn": syn,
                "layers": keep,
            })
            print(f"  {lbl:<24} layers={keep} syn={syn:,} "
                  f"acc={acc0:.2f}% -> {acc:.2f}%", flush=True)

    _write_csv(rows)
    _plot(rows)
    _summary(rows)


def _summary(rows):
    print("\n" + "=" * 78)
    print("  ZUSAMMENFASSUNG (Mittelwert ueber Seeds)")
    print("=" * 78)
    print(f"  {'Methode':<26}{'Synapsen':>12}{'Acc voll':>10}{'Acc nach':>10}"
          f"{'Delta':>8}")
    print("-" * 78)
    for key, _, _, lbl in METHODS:
        rs = [r for r in rows if r["method"] == key]
        a0 = np.mean([r["acc0"] for r in rs])
        ac = np.mean([r["acc"] for r in rs])
        sy = np.mean([r["syn"] for r in rs])
        print(f"  {lbl:<26}{sy:>10,.0f}{a0:>10.2f}{ac:>10.2f}{ac-a0:>+8.2f}")
    print("=" * 78)


def _plot(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    keys = [k for k, _, _, _ in METHODS]
    lbls = [l for _, _, _, l in METHODS]
    x = np.arange(len(keys))
    acc_mean = [np.mean([r["acc"] for r in rows if r["method"] == k]) for k in keys]
    acc0 = np.mean([r["acc0"] for r in rows])
    ax.bar(x, acc_mean, width=0.55, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.axhline(acc0, color="black", ls="--", lw=1, label=f"Voll (vor Prune) {acc0:.2f}%")
    for i, v in enumerate(acc_mean):
        ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=15, ha="right")
    ax.set_ylabel("Test-Accuracy nach Pruning (%)")
    ax.set_title(f"Prune-Methoden, Budget {SYN_TARGET:,} Synapsen (Mittelwert)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    png = os.path.join(_HERE, "RESULT_bild_prune_methods.png")
    fig.savefig(png, dpi=150)
    print(f"  Plot gespeichert: {png}", flush=True)


def _write_csv(rows):
    csvp = os.path.join(_HERE, "RESULT_prune_methods_compare.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "method", "label", "acc0", "acc", "syn0", "syn",
                    "layer1", "layer2", "layer3"])
        for r in rows:
            w.writerow([r["seed"], r["method"], r["label"],
                        round(r["acc0"], 3), round(r["acc"], 3),
                        r["syn0"], r["syn"]] + r["layers"])
    print(f"  CSV gespeichert: {csvp}", flush=True)


if __name__ == "__main__":
    main()
