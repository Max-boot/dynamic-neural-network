"""
Faire Vergleich: Meta-Scorer (zero-shot) vs. Baseline auf den gleichen
Zielgroessen wie der bestehende Sweep.
================================================================
Fragestellung: Wie schlaegt sich der EINMALIG meta-trainierte Scorer
(models/neuron_scorer_meta.pt), wenn er - OHNE Retraining - auf dem
Standard-Layout [256,128,64] ueber die gleichen 8 Zielgroessen wie der
bestehende Sweep angewendet wird?

Beide Varianten (Baseline = Aktivierungs-Pruning, Meta = geladener
Scorer) laufen hier NEU und unter IDENTISCHEN Bedingungen:
  - gleiches Netz [256,128,64]
  - gleiche Seeds (42, 2024)
  - gleiche 8 Zielgroessen
  - gleiche Epochenzahl, gleiches Prune-Scheduling (enforce)
  - der Meta-Scorer wird NICHT neu trainiert (zero-shot)

Ausgabe:
  - Konsolen-Tabelle (acc vor/nach, Delta, Synapsen)
  - Pareto-Graph sweep_metascorer_compare.png
  - Rohdaten sweep_metascorer_compare.csv
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, sys, importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

torch.manual_seed(42)
np.random.seed(42)

# ── Wiederverwendung der Meta-Trainings-Kernfunktionen ────────────────────
def _load_mt():
    spec = importlib.util.spec_from_file_location(
        "mt", r"D:\Dynamic Neural Networt (DNN)\Code\structural_ml_metatrain.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt

mt = _load_mt()
device = mt.device
train_loader = mt.train_loader
test_loader = mt.test_loader
train_ds = mt.train_ds
test_ds = mt.test_ds
print(f"Device: {device}", flush=True)

# Der Meta-Scorer wird geladen (optional vorab prozess-weit genutzt)
def load_meta_scorer(path="models/neuron_scorer_meta.pt"):
    scorer = mt.NeuronScorer(n_feat=6).to(device)
    scorer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    scorer.eval()
    return scorer


# ── Zielgroessen + Seeds (identisch zum bestehenden Sweep) ───────────────
TARGETS = [
    [256, 128, 64],
    [224, 112, 56],
    [192, 96, 48],
    [160, 80, 40],
    [128, 64, 32],
    [96, 48, 24],
    [64, 32, 16],
    [48, 24, 12],
]
SEEDS = [42, 2024]
LAYOUT = [256, 128, 64]


def run_config(use_meta, seed, target_neurons, scorer, warmup_epochs=3,
               epochs=16, prune_interval=2):
    """Eine Konfiguration. use_meta=True -> geladener Scorer (zero-shot)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = mt.DynamicMLP(LAYOUT).to(device)
    syn_start = model.active_synapses
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    prune_epochs = [ep for ep in range(1, epochs + 1) if ep % prune_interval == 0]
    total_prune_steps = len(prune_epochs)
    step = 0
    start_neurons = list(LAYOUT)
    acc_after_last = None

    # ── Warmup ───────────────────────────────────────────────────────────
    for _ in range(warmup_epochs):
        model.train()
        model.reset_activity_buffer()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    acc_start = mt.measure_accuracy(model)

    # ── Pruning-Phase ────────────────────────────────────────────────────
    accs = []
    for ep in range(epochs):
        model.train()
        model.reset_activity_buffer()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        acc = mt.measure_accuracy(model)
        keep = list(model.neuron_counts)
        if (ep + 1) % prune_interval == 0 and total_prune_steps > 0:
            step += 1
            progress = min(1.0, step / total_prune_steps)
            new_counts = []
            for s0, tgt in zip(start_neurons, target_neurons):
                new_counts.append(int(round(s0 + (tgt - s0) * progress)))
            new_counts = [int(min(max(n, t), c)) for n, t, c in
                          zip(new_counts, target_neurons, start_neurons)]
            if sum(new_counts) < sum(model.neuron_counts):
                keep = list(target_neurons) if step >= total_prune_steps \
                    else new_counts

                # Aktivierungen fuer Pruning in Puffer sammeln
                model.reset_activity_buffer()
                if use_meta:
                    model.reset_class_stats()
                    for xb, yb in list(train_loader)[:3]:
                        model(xb.to(device))
                        model.update_class_stats(xb.to(device), yb.to(device))
                    corr = model.compute_correlation_matrix()
                    feats = mt.extract_features(model, corr)
                    with torch.no_grad():
                        score = scorer(
                            torch.from_numpy(feats).float().to(device)) \
                            .cpu().numpy()
                    model.apply_neuron_pruning(keep, scores_all=score)
                    model.reset_class_stats()
                else:
                    model(xb0 := next(iter(train_loader))[0].to(device))
                    model.apply_neuron_pruning(keep)

        accs.append(acc)

    return {
        "acc_start": acc_start,
        "acc_end": accs[-1],
        "neurons_end": model.neuron_counts,
        "syn_start": syn_start,
        "syn_end": model.active_synapses,
    }


def main():
    quick = "--quick" in sys.argv
    if quick:
        targets = [TARGETS[4]]
        seeds = [SEEDS[0]]
    else:
        targets = TARGETS
        seeds = SEEDS
    scorer = load_meta_scorer()
    print("  Meta-Scorer geladen (models/neuron_scorer_meta.pt), zero-shot",
          flush=True)

    results = {"baseline": [], "meta": []}
    for ti, tgt in enumerate(targets):
        for seed in seeds:
            for use_meta, key in ((False, "baseline"), (True, "meta")):
                r = run_config(use_meta, seed, tgt, scorer)
                r["target"] = tgt
                r["seed"] = seed
                results[key].append(r)
                print(f"  [{ti+1}/{len(TARGETS)}] {key:>8} {tgt} seed={seed} "
                      f"acc_end={r['acc_end']:.2f}% syn_end={r['syn_end']:,}",
                      flush=True)

    report(results)


def report(results):
    print("\n" + "=" * 108)
    print("  VERGLEICH Meta-Scorer (zero-shot) vs. Baseline - gleiche Zielgroessen")
    print("=" * 108)
    print(f"  {'Zielgroesse':<16}{'Typ':<10}{'Acc start':>10}{'Acc end':>10}"
          f"{'Delta':>8}{'Syn end':>11}")
    print("-" * 108)
    # nur Zielgroessen, die tatsaechlich vorhanden sind (robust gegen --quick)
    present = sorted({tuple(r["target"]) for r in results["baseline"]}
                     | {tuple(r["target"]) for r in results["meta"]},
                     key=lambda t: sum(t), reverse=True)
    table = []
    for tgt in present:
        row = {"target": tgt}
        for key in ("baseline", "meta"):
            rs = [r for r in results[key] if tuple(r["target"]) == tuple(tgt)]
            row[f"{key}_acc"] = np.mean([r["acc_end"] for r in rs])
            row[f"{key}_syn"] = np.mean([r["syn_end"] for r in rs])
            row[f"{key}_start"] = np.mean([r["acc_start"] for r in rs])
            print(f"  {str(tgt):<16}{key:<10}"
                  f"{row[f'{key}_start']:>9.2f}%{row[f'{key}_acc']:>9.2f}%"
                  f"{row[f'{key}_acc'] - row[f'{key}_start']:>+7.2f}"
                  f"{int(row[f'{key}_syn']):>11,}")
        table.append(row)
    print("=" * 108)

    # Pareto-Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, marker, label in (
            ("baseline", "#1f77b4", "o", "Baseline (Aktivierungs-Pruning)"),
            ("meta", "#d62728", "s", "Meta-Scorer (zero-shot)")):
        xs = [r["syn_end"] for r in results[key]]
        ys = [r["acc_end"] for r in results[key]]
        ax.scatter(xs, ys, color=color, marker=marker, s=40, alpha=0.55,
                   label=f"{label} (einzelne Seeds)")
        seen = {}
        for r in results[key]:
            seen.setdefault(tuple(r["target"]), []).append(r)
        mx = [np.mean([q["syn_end"] for q in v]) for v in seen.values()]
        my = [np.mean([q["acc_end"] for q in v]) for v in seen.values()]
        order = np.argsort(mx)
        ax.plot([mx[i] for i in order], [my[i] for i in order],
                color=color, lw=2.5,
                label=f"{key} (Mittelwert)")
        for (k, v) in seen.items():
            cx = np.mean([q["syn_end"] for q in v])
            cy = np.mean([q["acc_end"] for q in v])
            ax.annotate(f"{list(k)}", (cx, cy), textcoords="offset points",
                        xytext=(0, 6), fontsize=7, color=color, ha="center")
    ax.set_xlabel("Synapsen nach Pruning")
    ax.set_ylabel("Test-Accuracy nach Pruning (%)")
    ax.set_title("Baseline vs. Meta-Scorer auf gleichen Zielgroessen (Layout [256,128,64])")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("sweep_metascorer_compare.png", dpi=150)
    print("  Plot gespeichert: sweep_metascorer_compare.png", flush=True)

    # CSV
    with open("sweep_metascorer_compare.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["target", "baseline_acc", "baseline_syn",
                     "meta_acc", "meta_syn"])
        for row in table:
            wr.writerow([row["target"], round(row["baseline_acc"], 3),
                         int(row["baseline_syn"]),
                         round(row["meta_acc"], 3), int(row["meta_syn"])])
    print("  CSV gespeichert: sweep_metascorer_compare.csv", flush=True)


if __name__ == "__main__":
    main()
