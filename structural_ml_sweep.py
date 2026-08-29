"""
Structural ML Controller — Parameter-Sweep über verschiedene Zielgrößen.
========================================================================
Untersucht, wie sich ein externer ML-Kontroller (trainierter NeuronScorer,
"WO") + acc-bewusstes Gate ("WANN") im Vergleich zum Fallback-Pruning
über mehrere Zielgrößen (Neuronen/Synapsen) verhält.

Pro Zielgröße werden beide Konfigurationen (Baseline vs. ML) mit mehreren
Seeds trainiert. Am Ende wird ein Pareto-Graph erzeugt:
  X-Achse: erreichte Synapsenzahl (bzw. Neuronen)
  Y-Achse: Test-Accuracy nach dem Pruning
Baseline- und ML-Kurve werden überlagert -> Trade-off visualisiert.

Läuft auf GPU, falls CUDA verfügbar (sonst CPU).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ── MNIST ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./mnist", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=2048, shuffle=False)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1) Netzwerk (identisch zu v2)
# ═══════════════════════════════════════════════════════════════════════════
class MLPWithStruktur(nn.Module):
    HIDDEN_SIZES = (256, 128, 64)

    def __init__(self):
        super().__init__()
        sizes = self.HIDDEN_SIZES
        self.fc1 = nn.Linear(28 * 28, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], sizes[2])
        self.fc4 = nn.Linear(sizes[2], 10)

        for i, s in enumerate(sizes):
            self.register_buffer(f"neuron_mask_{i}",
                                 torch.ones(s, dtype=torch.float32))
        self._act_buffer = []
        self.register_buffer("class_act_sum",
                             torch.zeros((10, self.gap_tot_hidden)))
        self.register_buffer("class_act_count", torch.zeros(10))

    @property
    def neuron_masks(self):
        return [getattr(self, f"neuron_mask_{i}")
                for i in range(len(self.HIDDEN_SIZES))]

    @property
    def hidden_start_indices(self):
        out, acc = [], 0
        for sz in self.HIDDEN_SIZES:
            out.append(acc)
            acc += sz
        return out

    @property
    def gap_tot_hidden(self):
        return int(sum(self.HIDDEN_SIZES))

    @property
    def neuron_counts(self):
        return [int(m.sum().item()) for m in self.neuron_masks]

    @property
    def active_synapses(self):
        wp = [self.fc1.weight, self.fc2.weight, self.fc3.weight, self.fc4.weight]
        c = self.neuron_counts
        return (wp[0].shape[1] * c[0] + c[0] * c[1] + c[1] * c[2] + c[2] * 10)

    def reset_activity_buffer(self):
        self._act_buffer = []

    def reset_class_stats(self):
        self.class_act_sum.zero_()
        self.class_act_count.zero_()

    def forward(self, x, record=True):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)) * self.neuron_masks[0].unsqueeze(0)
        h2 = F.relu(self.fc2(h1)) * self.neuron_masks[1].unsqueeze(0)
        h3 = F.relu(self.fc3(h2)) * self.neuron_masks[2].unsqueeze(0)
        out = self.fc4(h3)
        if record:
            self._act_buffer.append(
                torch.cat([h1.detach(), h2.detach(), h3.detach()], dim=1))
        return out

    def forward_repr(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)) * self.neuron_masks[0].unsqueeze(0)
        h2 = F.relu(self.fc2(h1)) * self.neuron_masks[1].unsqueeze(0)
        h3 = F.relu(self.fc3(h2)) * self.neuron_masks[2].unsqueeze(0)
        return self.fc4(h3), h1, h2, h3

    def update_class_stats(self, x, y):
        with torch.no_grad():
            _, h1, h2, h3 = self.forward_repr(x)
            ha = torch.cat([h1, h2, h3], dim=1).detach()
            yc = y
            for d in range(10):
                m = yc == d
                if m.any():
                    self.class_act_sum[d] += ha[m].sum(dim=0)
                    self.class_act_count[d] += m.sum()

    def class_specificity(self):
        d = self.class_act_sum.device
        spec = torch.zeros(self.gap_tot_hidden, device=d)
        for i, sz in enumerate(self.HIDDEN_SIZES):
            lo = self.hidden_start_indices[i]
            hi = lo + sz
            counts = self.class_act_count.clone()
            counts[counts == 0] = 1.0
            cond_mean = (self.class_act_sum / counts.view(-1, 1))[:, lo:hi]
            gmean = cond_mean.mean(dim=0)
            gstd = cond_mean.std(dim=0) + 1e-8
            dev = (cond_mean - gmean) / gstd
            spec[lo:hi] = dev.abs().max(dim=0).values
        return spec.cpu()

    def compute_correlation_matrix(self):
        if len(self._act_buffer) == 0:
            return np.zeros((self.gap_tot_hidden, self.gap_tot_hidden))
        acts = torch.cat(self._act_buffer, dim=0).cpu().numpy()
        if acts.shape[0] > 4096:
            acts = acts[np.random.choice(acts.shape[0], 4096, replace=False)]
        a = acts - acts.mean(axis=0, keepdims=True)
        sd = a.std(axis=0)
        sd[sd < 1e-8] = 1e-8
        corr = np.nan_to_num((a.T @ a) / a.shape[0])
        return corr

    def apply_neuron_pruning(self, keep_per_layer, scores_all=None):
        acts = torch.cat(self._act_buffer, dim=0).cpu()
        dev = next(self.parameters()).device
        starts = self.hidden_start_indices
        for i, sz in enumerate(self.HIDDEN_SIZES):
            keep = int(keep_per_layer[i])
            layer_acts = acts[:, starts[i]:starts[i] + sz].mean(dim=0)
            if scores_all is None:
                rank = torch.argsort(layer_acts, descending=True)
            else:
                if not torch.is_tensor(scores_all):
                    scores_all = torch.from_numpy(np.asarray(scores_all)).float()
                rank = torch.argsort(scores_all[starts[i]:starts[i] + sz],
                                     descending=True)
            new_mask = torch.zeros(sz, dtype=torch.float32)
            new_mask[rank[:keep]] = 1.0
            setattr(self, f"neuron_mask_{i}", new_mask.to(dev))


# ═══════════════════════════════════════════════════════════════════════════
# 2) Features + Gradient-Wichtigkeit
# ═══════════════════════════════════════════════════════════════════════════
def compute_batch_importance(model, x, y):
    x = x.to(device)
    y = y.to(device)
    x.requires_grad_(True)
    out, h1, h2, h3 = model.forward_repr(x)
    loss = F.cross_entropy(out, y)
    g1, g2, g3 = torch.autograd.grad(loss, (h1, h2, h3))
    imp = torch.cat([(h1 * g1).abs().mean(dim=0),
                     (h2 * g2).abs().mean(dim=0),
                     (h3 * g3).abs().mean(dim=0)])
    return imp.detach().cpu()


def extract_features(model, corr):
    n = model.gap_tot_hidden
    acts = torch.cat(model._act_buffer, dim=0).cpu() if model._act_buffer else \
        torch.zeros(1, n)
    act_mean = acts.mean(dim=0)
    part_rate = (acts > 0).float().mean(dim=0)
    spec = model.class_specificity()
    starts = model.hidden_start_indices
    sizes = model.HIDDEN_SIZES

    mean_corr = np.zeros(n)
    max_corr = np.zeros(n)
    for i, sz in enumerate(sizes):
        lo = starts[i]
        hi = lo + sz
        sub = np.abs(corr[lo:hi, lo:hi])
        np.fill_diagonal(sub, 0.0)
        mean_corr[lo:hi] = sub.mean(axis=1)
        max_corr[lo:hi] = sub.max(axis=1)

    layer_idx = np.zeros(n)
    for i, sz in enumerate(sizes):
        layer_idx[starts[i]:starts[i] + sz] = i / max(1, len(sizes) - 1)

    spec = spec.numpy()
    spec = spec / (spec.max() + 1e-8)
    act_mean = act_mean.numpy()
    act_mean = act_mean / (act_mean.max() + 1e-8)
    mean_corr = mean_corr / (mean_corr.max() + 1e-8)
    max_corr = max_corr / (max_corr.max() + 1e-8)

    feats = np.stack([act_mean, part_rate.numpy(), mean_corr, max_corr,
                      spec, layer_idx], axis=1)
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# 3) ML-Kontroller
# ═══════════════════════════════════════════════════════════════════════════
class NeuronScorer(nn.Module):
    def __init__(self, n_feat=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StrukturControllerML:
    def __init__(self, target_neurons, total_prune_steps=6, prune_interval=2,
                 acc_tolerance=100.0, enforce=True):
        self.target_neurons = list(target_neurons)
        self.total_prune_steps = total_prune_steps
        self.prune_interval = prune_interval
        self.acc_tolerance = acc_tolerance
        self.enforce = enforce
        self.scorer = NeuronScorer(n_feat=6).to(device)
        self.trained = False
        self.history = []
        self.start_neurons = None
        self._prune_step = 0
        self.acc_after_last_prune = None

    def train_scorer_on_data(self, X, Y, epochs=300):
        Xt = torch.from_numpy(X).float().to(device)
        Yt = torch.from_numpy(Y).float().to(device)
        w = (Yt + 1e-3)
        opt = torch.optim.Adam(self.scorer.parameters(), lr=1e-2)
        self.scorer.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.scorer(Xt)
            loss = ((pred - Yt) ** 2 * w).mean()
            loss.backward()
            opt.step()
        self.trained = True

    def decide(self, model, corr, current_neurons, epoch, last_acc):
        keep_per_layer = list(current_neurons)
        action = "none"
        if self.start_neurons is None:
            self.start_neurons = list(current_neurons)

        do_prune = epoch >= 1 and epoch % self.prune_interval == 0
        if do_prune and self.total_prune_steps > 0:
            skip = (not self.enforce
                    and self.acc_after_last_prune is not None
                    and self.acc_after_last_prune < last_acc - self.acc_tolerance)
            if skip:
                action = "skip (recover)"
            else:
                self._prune_step += 1
                progress = min(1.0, self._prune_step / self.total_prune_steps)
                new_counts = []
                for start, tgt in zip(self.start_neurons, self.target_neurons):
                    target_here = start + (tgt - start) * progress
                    new_counts.append(int(round(target_here)))
                new_counts = [int(min(max(n, t), c)) for n, t, c in
                              zip(new_counts, self.target_neurons, self.start_neurons)]
                if sum(new_counts) < sum(current_neurons):
                    keep_per_layer = new_counts
                    action = "prune"
                    if self._prune_step >= self.total_prune_steps:
                        keep_per_layer = list(self.target_neurons)

        result = {
            "action": action,
            "keep_per_layer": keep_per_layer,
            "current_neurons": list(current_neurons),
            "epoch": epoch,
            "avg_abs_corr": float(np.abs(corr).mean()) if corr is not None else 0.0,
        }
        self.history.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 4) Trainings-Lauf (parametrisiert über Zielgröße + Seed)
# ═══════════════════════════════════════════════════════════════════════════
def measure_accuracy(model):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x, record=False).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def run_config(use_ml, seed, target_neurons, warmup_epochs=3, epochs=16,
               prune_interval=2, acc_tolerance=100.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPWithStruktur().to(device)
    syn_start = model.active_synapses
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Anzahl geplanter Prune-Schritte über die Pruning-Phase
    total_prune_steps = len([ep for ep in range(1, epochs + 1)
                             if ep % prune_interval == 0])

    # ── Warmup ───────────────────────────────────────────────────────────
    for ep in range(warmup_epochs):
        model.train()
        model.reset_activity_buffer()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    acc_start = measure_accuracy(model)

    controller = StrukturControllerML(target_neurons=target_neurons,
                                      total_prune_steps=total_prune_steps,
                                      prune_interval=prune_interval,
                                      acc_tolerance=acc_tolerance)

    # ── Scorer-Training (nur ML) ─────────────────────────────────────────
    if use_ml:
        model.train()
        sampler = iter(train_loader)
        all_feats, all_imp = [], []
        for _ in range(6):
            try:
                xb, yb = next(sampler)
            except StopIteration:
                sampler = iter(train_loader)
                xb, yb = next(sampler)
            xb, yb = xb.to(device), yb.to(device)
            model.update_class_stats(xb, yb)
            imp = compute_batch_importance(model, xb, yb).numpy()
            model.reset_activity_buffer()
            model(xb)
            corr = model.compute_correlation_matrix()
            feats = extract_features(model, corr)
            all_feats.append(feats)
            all_imp.append(imp)
        X_all = np.concatenate(all_feats, axis=0)
        Y_all = np.concatenate(all_imp, axis=0)
        starts = model.hidden_start_indices
        for i, sz in enumerate(model.HIDDEN_SIZES):
            lo, hi = starts[i], starts[i] + sz
            Y_all[lo:hi] = Y_all[lo:hi] / (Y_all[lo:hi].max() + 1e-8)
        controller.train_scorer_on_data(X_all, Y_all)

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

        acc = measure_accuracy(model)
        corr = model.compute_correlation_matrix()
        decision = controller.decide(model, corr, model.neuron_counts,
                                     ep + 1, acc)

        if decision["action"] == "prune":
            if use_ml and controller.trained:
                feats = extract_features(model, corr)
                Xt = torch.from_numpy(feats).float().to(device)
                controller.scorer.eval()
                with torch.no_grad():
                    score = controller.scorer(Xt).cpu().numpy()
                model.apply_neuron_pruning(decision["keep_per_layer"],
                                           scores_all=score)
            else:
                model.apply_neuron_pruning(decision["keep_per_layer"])
            controller.acc_after_last_prune = measure_accuracy(model)
            acc = controller.acc_after_last_prune

        accs.append(acc)

    return {
        "acc_start": acc_start,
        "acc_end": accs[-1],
        "neurons_start": list(MLPWithStruktur.HIDDEN_SIZES),
        "neurons_end": model.neuron_counts,
        "syn_start": syn_start,
        "syn_end": model.active_synapses,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5) Sweep
# ═══════════════════════════════════════════════════════════════════════════
# 8 Zielgrößen (Verhältnis 2:1:1 wie das Originalnetz), absteigend
TARGETS = [
    [256, 128, 64],   # kein Pruning -> Referenz (volles Netz)
    [224, 112, 56],
    [192, 96, 48],
    [160, 80, 40],
    [128, 64, 32],
    [96, 48, 24],
    [64, 32, 16],
    [48, 24, 12],
]
SEEDS = [42, 2024]


def synapsen(target):
    c256, c128, c64, c10 = 256, 128, 64, 10
    return 784 * target[0] + target[0] * target[1] + target[1] * target[2] + \
        target[2] * c10


def main(n_targets=None, n_seeds=None, quick=False):
    targets = TARGETS if n_targets is None else TARGETS[:n_targets]
    seeds = SEEDS if n_seeds is None else SEEDS[:n_seeds]
    if quick:
        targets = [TARGETS[4]]
        seeds = [SEEDS[0]]
    print("\n" + "=" * 72)
    print(f"  SWEEP: {len(targets)} Zielgroessen x {len(seeds)} Seeds x "
          f"(Baseline + ML)")
    print("=" * 72, flush=True)

    results = {"baseline": [], "ml": []}

    for ti, tgt in enumerate(targets):
        for seed in seeds:
            for use_ml in (False, True):
                kind = "ml" if use_ml else "baseline"
                r = run_config(use_ml=use_ml, seed=seed, target_neurons=tgt)
                r["target"] = tgt
                r["seed"] = seed
                results[kind].append(r)
                print(f"  [{ti+1}/{len(targets)}] {kind:>8} "
                      f"target={tgt} seed={seed} "
                      f"acc_end={r['acc_end']:.2f}% "
                      f"neurons_end={r['neurons_end']} "
                      f"syn_end={r['syn_end']:,}", flush=True)

    build_plots(results)

    # Konsolen-Tabelle
    print("\n" + "=" * 100)
    print("  UEBERSICHT (Mittelwert ueber Seeds)")
    print("=" * 100)
    print(f"  {'Zielgroesse':<18}{'Typ':<10}{'Acc start':>10}{'Acc end':>10}"
          f"{'Delta Acc':>10}{'Synapsen end':>14}")
    print("-" * 100)
    for tgt in TARGETS:
        for kind in ("baseline", "ml"):
            rs = [r for r in results[kind] if tuple(r["target"]) == tuple(tgt)]
            accs = [r["acc_end"] for r in rs]
            starts = [r["acc_start"] for r in rs]
            syns = [r["syn_end"] for r in rs]
            mean_acc = np.mean(accs)
            mean_start = np.mean(starts)
            print(f"  {str(tgt):<18}{kind:<10}{mean_start:>9.2f}%"
                  f"{mean_acc:>9.2f}%{mean_acc - mean_start:>+7.2f}"
                  f"{int(np.mean(syns)):>14,}")
    print("=" * 100)


def build_plots(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def scatter(ax, kind, color, label):
        xs = [r["syn_end"] for r in results[kind]]
        ys = [r["acc_end"] for r in results[kind]]
        ax.scatter(xs, ys, color=color, label=label, s=28, alpha=0.7)
        # Mittelwert pro Zielgröße
        seen = {}
        for r in results[kind]:
            key = tuple(r["target"])
            seen.setdefault(key, []).append(r)
        mx = [np.mean([q["syn_end"] for q in v]) for v in seen.values()]
        my = [np.mean([q["acc_end"] for q in v]) for v in seen.values()]
        order = np.argsort(mx)
        ax.plot([mx[i] for i in order], [my[i] for i in order],
                color=color, lw=2, alpha=0.9)
        return xs, ys

    # Panel 1: Accuracy vs. Synapsen (Pareto)
    ax = axes[0]
    b = scatter(ax, "baseline", "#1f77b4", "Baseline (Fallback)")
    m = scatter(ax, "ml", "#d62728", "ML-Kontroller")
    # Beschriftung der Mittelwerte
    for kind, color in (("baseline", "#1f77b4"), ("ml", "#d62728")):
        seen = {}
        for r in results[kind]:
            seen.setdefault(tuple(r["target"]), []).append(r)
        for key, v in seen.items():
            mx = np.mean([q["syn_end"] for q in v])
            my = np.mean([q["acc_end"] for q in v])
            ax.annotate(f"{key}", (mx, my), textcoords="offset points",
                        xytext=(0, 6), fontsize=7, color=color, ha="center")
    ax.set_xlabel("Synapsen (Anzahl aktiver Gewichte)")
    ax.set_ylabel("Test-Accuracy nach Pruning (%)")
    ax.set_title("Pareto: Accuracy vs. Synapsen")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Accuracy vs. Neuronen
    ax = axes[1]
    for kind, color, label in (("baseline", "#1f77b4", "Baseline (Fallback)"),
                               ("ml", "#d62728", "ML-Kontroller")):
        seen = {}
        for r in results[kind]:
            seen.setdefault(tuple(r["target"]), []).append(r)
        mx = [np.mean([q["syn_end"] for q in v]) for v in seen.values()]
        my = [np.mean([sum(q["neurons_end"]) for q in v]) for v in seen.values()]
        accs = [np.mean([q["acc_end"] for q in v]) for v in seen.values()]
        order = np.argsort(mx)
        ax.plot([my[i] for i in order], [accs[i] for i in order],
                color=color, lw=2, label=label, marker="o", alpha=0.9)
    ax.set_xlabel("Neuronen (Ende, gesamt)")
    ax.set_ylabel("Test-Accuracy nach Pruning (%)")
    ax.set_title("Accuracy vs. Neuronen (Ende)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig("sweep_result.png", dpi=150)
    print("\n  Graph gespeichert: sweep_result.png", flush=True)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
