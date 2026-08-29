"""
Meta-Training eines layout-invarianten NeuronScorer (Option 2).
================================================================
Ziel: EIN einziger trainierter ML-Kontroller (NeuronScorer), der ueber
BELIEBIGE MLP-Architekturen (verschiedene Tiefe und Breite) generalisiert,
ohne fuer jedes Layout neu trainiert werden zu muessen.

Was geaendert wird, um Layout-Invarianz zu erreichen:

  1) LABEL-Normalisierung (invariant):
     Statt absoluter Salienz (je Layout auf max normiert -> layout-sensitiv)
     wird die OBD-Salienz pro SCHICHT in einen Rang (0..1) umgewandelt.
     Das wichtigste Neuron einer Schicht bekommt 1, das unwichtigste 0.
     Ein Rang ist voellig unabhaengig von der Schichtgroesse -> generalisiert
     ueber verschiedene Layouts.

  2) FEATURE-Normalisierung (lokal pro Schicht):
     Aktivierungs-, Korrelations- und Spezifitaets-Merkmale werden pro Schicht
     auf [0,1] normiert (relativ zu den Schicht-Kameraden). Dadurch sind die
     Eingabeverteilungen vergleichbar, egal wie breit/tief das Netz ist.

Meta-Training:
  - Mehrere TRAIN-Layouts (verschiedene Architekturen) werden kurz trainiert.
  - Aus jedem Layout werden viele (Feature, Label)-Samples extrahiert.
  - Alle Samples werden gemischt und EIN NeuronScorer ausfuehrlich trainiert.
  - Der Scorer wird gespeichert (neuron_scorer_meta.pt).

Zero-shot-Test:
  - Ein UNGESEHENES Layout (nie im Training) wird trainiert und mit dem
    gespeicherten Scorer gepruntet (ohne jedes Neulernen).
  - Vergleich von Accuracy + erreichten Synapsen mit der Baseline:
    beweist, ob der ML-Kontroller auf neue Layouts generalisiert.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

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
test_loader  = DataLoader(test_ds, batch_size=4096, shuffle=False)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1) Generisches MLP mit variabler Tiefe/Breite + Struktur-Methoden
# ═══════════════════════════════════════════════════════════════════════════
class DynamicMLP(nn.Module):
    """MLP mit beliebiger versteckter Architektur (Tiefe/Breite variabel)."""

    def __init__(self, hidden_sizes, n_classes=10, input_dim=28 * 28):
        super().__init__()
        self.HIDDEN_SIZES = tuple(hidden_sizes)
        self.n_layers = len(self.HIDDEN_SIZES)
        self.input_dim = input_dim
        self.n_classes = n_classes

        dims = [input_dim] + list(self.HIDDEN_SIZES) + [n_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        for i, s in enumerate(self.HIDDEN_SIZES):
            self.register_buffer(f"neuron_mask_{i}", torch.ones(s, dtype=torch.float32))
        self._act_buffer = []
        self.register_buffer("class_act_sum",
                             torch.zeros((n_classes, self.gap_tot_hidden)))
        self.register_buffer("class_act_count", torch.zeros(n_classes))

    # ── Hidden-Infrastruktur ─────────────────────────────────────────────
    @property
    def neuron_masks(self):
        return [getattr(self, f"neuron_mask_{i}") for i in range(self.n_layers)]

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
        c = self.neuron_counts
        dims = [self.input_dim] + c + [self.n_classes]
        s = 0
        for i in range(len(dims) - 1):
            s += dims[i] * dims[i + 1]
        return int(s)

    def reset_activity_buffer(self):
        self._act_buffer = []

    def reset_class_stats(self):
        self.class_act_sum.zero_()
        self.class_act_count.zero_()

    # ── Forward ──────────────────────────────────────────────────────────
    def _hidden(self, x):
        """Liefert (out, verdeckte Aktivierungsliste)."""
        h = x
        if h.dim() > 2:
            h = h.view(h.size(0), -1)
        acts = []
        for i in range(self.n_layers):
            h = F.relu(self.layers[i](h)) * self.neuron_masks[i].unsqueeze(0)
            acts.append(h)
        out = self.layers[self.n_layers](h)
        return out, acts

    def forward(self, x, record=True):
        out, acts = self._hidden(x)
        if record:
            self._act_buffer.append(torch.cat([a.detach() for a in acts], dim=1))
        return out

    def forward_repr(self, x):
        out, acts = self._hidden(x)
        return out, acts

    def update_class_stats(self, x, y):
        with torch.no_grad():
            _, acts = self.forward_repr(x)
            ha = torch.cat(acts, dim=1).detach()
            yc = y
            for d in range(self.n_classes):
                mm = yc == d
                if mm.any():
                    self.class_act_sum[d] += ha[mm].sum(dim=0)
                    self.class_act_count[d] += mm.sum()

    def class_specificity(self):
        d = self.class_act_sum.device
        spec = torch.zeros(self.gap_tot_hidden, device=d)
        starts = self.hidden_start_indices
        for i, sz in enumerate(self.HIDDEN_SIZES):
            lo = starts[i]
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
        n = self.gap_tot_hidden
        if len(self._act_buffer) == 0:
            return np.zeros((n, n))
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
# 2) Layout-invariante Labels + lokal normalisierte Features
# ═══════════════════════════════════════════════════════════════════════════
def compute_batch_importance(model, x, y):
    """OBD-Salienz |a*grad_L| je Neuron (Tensor der Laenge tot_hidden)."""
    x = x.to(device)
    y = y.to(device)
    x.requires_grad_(True)
    out, acts = model.forward_repr(x)
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, acts)
    imp = torch.cat([(a * g).abs().mean(dim=0) for a, g in zip(acts, grads)])
    return imp.detach().cpu()


def to_per_layer_rank(values, model):
    """
    Konvertiert absolute Werte in einen Rangwert 0..1 pro Schicht.
    Unabhaengig von der Schichtgroesse -> layout-invariant.
    :return: Ranking 0..1 (Wichtigstes Neuron einer Schicht = 1)
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    out = np.zeros_like(v)
    starts = model.hidden_start_indices
    for i, sz in enumerate(model.HIDDEN_SIZES):
        lo = starts[i]
        hi = lo + sz
        seg = v[lo:hi]
        order = np.argsort(np.argsort(seg))          # Rang 0..sz-1
        if sz > 1:
            out[lo:hi] = order / (sz - 1.0)          # 0..1
    return out


def normalize_locally(feat, model):
    """Normiert ein Feature pro Schicht auf [0,1] (min-max ueber Schicht)."""
    f = np.asarray(feat, dtype=np.float64).ravel()
    out = np.zeros_like(f)
    starts = model.hidden_start_indices
    for i, sz in enumerate(model.HIDDEN_SIZES):
        lo = starts[i]
        hi = lo + sz
        seg = f[lo:hi]
        mn, mx = seg.min(), seg.max()
        out[lo:hi] = (seg - mn) / (mx - mn + 1e-8)
    return out


def extract_features(model, corr, n_layers_ref=None):
    """
    Layout-invariante Features pro Neuron (6 Merkmale, alle lokal normalisiert).
    """
    n = model.gap_tot_hidden
    acts = torch.cat(model._act_buffer, dim=0).cpu() if model._act_buffer else \
        torch.zeros(1, n)
    act_mean = acts.mean(dim=0).numpy()
    part_rate = (acts > 0).float().mean(dim=0).numpy()
    spec = model.class_specificity().numpy()

    mean_corr = np.zeros(n)
    max_corr = np.zeros(n)
    starts = model.hidden_start_indices
    for i, sz in enumerate(model.HIDDEN_SIZES):
        lo = starts[i]
        hi = lo + sz
        sub = np.abs(corr[lo:hi, lo:hi])
        np.fill_diagonal(sub, 0.0)
        mean_corr[lo:hi] = sub.mean(axis=1)
        max_corr[lo:hi] = sub.max(axis=1)

    n_layers = model.n_layers
    layer_idx = np.zeros(n)
    for i, sz in enumerate(model.HIDDEN_SIZES):
        layer_idx[starts[i]:starts[i] + sz] = i / max(1, n_layers - 1)

    # Lokale Normalisierung pro Schicht (Layout-Invarianz)
    act_mean = normalize_locally(act_mean, model)
    mean_corr = normalize_locally(mean_corr, model)
    max_corr = normalize_locally(max_corr, model)
    spec = normalize_locally(spec, model)

    feats = np.stack([act_mean, part_rate, mean_corr, max_corr, spec,
                      layer_idx], axis=1)
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# 3) NeuronScorer (etwas groesser fuer Meta-Lernen)
# ═══════════════════════════════════════════════════════════════════════════
class NeuronScorer(nn.Module):
    def __init__(self, n_feat=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 4) Netz-Training + Datensatz-Erzeugung pro Layout
# ═══════════════════════════════════════════════════════════════════════════
def measure_accuracy(model, loader=None):
    loader = loader or test_loader
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x, record=False).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_network(model, epochs=4):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        model.reset_activity_buffer()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


def collect_samples(model, n_batches=12):
    """Extrahiert (X, Y)-Samples aus einem trainierten Layout."""
    Xs, Ys = [], []
    model.train()
    sampler = iter(train_loader)
    for _ in range(n_batches):
        try:
            x, y = next(sampler)
        except StopIteration:
            sampler = iter(train_loader)
            x, y = next(sampler)
        x, y = x.to(device), y.to(device)
        model.update_class_stats(x, y)
        imp = compute_batch_importance(model, x, y).numpy()
        model.reset_activity_buffer()
        model(x)
        corr = model.compute_correlation_matrix()
        feats = extract_features(model, corr)
        labels = to_per_layer_rank(imp, model)
        Xs.append(feats)
        Ys.append(labels)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


# ═══════════════════════════════════════════════════════════════════════════
# 5) Training des Scorers (meta, mit Validierung + Speichern)
# ═══════════════════════════════════════════════════════════════════════════
def train_meta_scorer(X_tr, Y_tr, X_va, Y_va, epochs=1500, batch=4096, lr=1e-3):
    scorer = NeuronScorer(n_feat=6).to(device)
    ds = TensorDataset(torch.from_numpy(X_tr).float(),
                       torch.from_numpy(Y_tr).float())
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(scorer.parameters(), lr=lr)
    w = torch.from_numpy(Y_tr + 1e-3).float().to(device)
    Xt = torch.from_numpy(X_tr).float().to(device)
    Yt = torch.from_numpy(Y_tr).float().to(device)
    Xv = torch.from_numpy(X_va).float().to(device)
    Yv = torch.from_numpy(Y_va).float().to(device)

    hist = {"loss": [], "val_loss": []}
    for ep in range(epochs):
        scorer.train()
        total, cnt = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = scorer(xb)
            loss = ((pred - yb) ** 2 * (yb + 1e-3)).mean()
            loss.backward()
            opt.step()
            total += loss.item()
            cnt += 1
        # volle Train-/Val-Loss fuer Monitoring
        scorer.eval()
        with torch.no_grad():
            pl = ((scorer(Xt) - Yt) ** 2 * w).mean().item()
            pv = ((scorer(Xv) - Yv) ** 2 * (Yv + 1e-3)).mean().item()
        hist["loss"].append(pl)
        hist["val_loss"].append(pv)
        if (ep + 1) % 250 == 0 or ep == 0:
            print(f"    [Scorer] ep {ep+1}/{epochs} "
                  f"loss={pl:.5f} val={pv:.5f}", flush=True)
    return scorer, hist


# ═══════════════════════════════════════════════════════════════════════════
# 6) Layouts
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_LAYOUTS = [
    [256, 128, 64],
    [128, 64],
    [192, 96, 48, 24],
    [320, 160],
    [96, 48, 32, 16, 8],
]
TEST_LAYOUTS = [
    [128, 128, 64],
    [160, 80],
    [64, 64, 64, 32],
]
LAYOUT_SEEDS = [0, 1]


def main(quick=False):
    train_layouts = [TRAIN_LAYOUTS[0]] if quick else TRAIN_LAYOUTS
    test_layouts = [TEST_LAYOUTS[0]] if quick else TEST_LAYOUTS

    print("\n" + "=" * 72)
    print("  PHASE 1: Meta-Datensatz aus TRAIN-Layouts erzeugen")
    print("=" * 72, flush=True)

    Xs, Ys, layouts = [], [], []
    for li, sizes in enumerate(train_layouts):
        for sd in LAYOUT_SEEDS:
            torch.manual_seed(sd)
            np.random.seed(sd)
            model = DynamicMLP(sizes).to(device)
            train_network(model, epochs=4)
            acc = measure_accuracy(model)
            X, Y = collect_samples(model, n_batches=12)
            Xs.append(X)
            Ys.append(Y)
            layouts.append(sizes)
            print(f"  [Layout {li+1}/{len(train_layouts)}] {sizes} "
                  f"(seed {sd}) acc={acc:.2f}% samples={X.shape[0]}",
                  flush=True)

    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)
    print(f"  Gesamt: {X_all.shape[0]} neuron Samples, {X_all.shape[1]} Features")

    # Split Train/Val (per-Sample, gemischt)
    perm = np.random.RandomState(123).permutation(X_all.shape[0])
    n_val = int(0.2 * X_all.shape[0])
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, Y_tr = X_all[tr_idx], Y_all[tr_idx]
    X_va, Y_va = X_all[va_idx], Y_all[va_idx]

    print("\n" + "=" * 72)
    print("  PHASE 2: Meta-Training des NeuronScorer")
    print("=" * 72, flush=True)

    scorer, hist = train_meta_scorer(
        X_tr, Y_tr, X_va, Y_va, epochs=300 if quick else 1500)

    os.makedirs("models", exist_ok=True)
    torch.save(scorer.state_dict(), "models/neuron_scorer_meta.pt")
    print("  Scorer gespeichert: models/neuron_scorer_meta.pt", flush=True)

    # ── Loss-Kurve ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist["loss"], label="Train-Loss")
    ax.plot(hist["val_loss"], label="Val-Loss")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("MSE (gewichtete Rang-Wichtigkeit)")
    ax.set_title("Meta-Training des NeuronScorer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("scorer_training_curve.png", dpi=150)
    print("  Loss-Kurve gespeichert: scorer_training_curve.png", flush=True)

    evaluate_zero_shot(scorer, test_layouts)


def evaluate_zero_shot(scorer, test_layouts):
    """
    Testet den trainierten Scorer auf UNGESEHENEN Layouts (zero-shot).
    Prunt mit dem gespeicherten ML (OHNE Neulernen des Scorers) und
    vergleicht Accuracy/Neuronen mit der Baseline.
    """
    print("\n" + "=" * 72)
    print("  PHASE 3: Zero-shot-Test auf ungesehenen Layouts")
    print("=" * 72, flush=True)
    rows = []
    for li, sizes in enumerate(test_layouts):
        for sd in LAYOUT_SEEDS:
            # ── ML-Variante ──────────────────────────────────────────────
            torch.manual_seed(sd)
            np.random.seed(sd)
            model = DynamicMLP(sizes).to(device)
            train_network(model, epochs=5)
            acc0_ml = measure_accuracy(model)
            syn0 = model.active_synapses
            keep = [max(2, s // 2) for s in sizes]   # halbe Groesse je Schicht
            model.reset_activity_buffer()
            sampler = iter(train_loader)
            xb, yb = next(sampler)
            model.update_class_stats(xb.to(device), yb.to(device))
            model.reset_activity_buffer()
            model(xb.to(device))
            corr = model.compute_correlation_matrix()
            feats = extract_features(model, corr)
            scorer.eval()
            with torch.no_grad():
                score = scorer(torch.from_numpy(feats).float().to(device)) \
                    .cpu().numpy()
            model.apply_neuron_pruning(keep, scores_all=score)
            # feinjustieren (nur das Netz, KEIN Scorer-Retraining)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(6):
                model.train()
                model.reset_activity_buffer()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    opt.step()
            acc_ml = measure_accuracy(model)
            syn_ml = model.active_synapses

            # ── Baseline-Variante (gleiche Groesse, Aktivierungs-Pruning)
            torch.manual_seed(sd)
            np.random.seed(sd)
            modelB = DynamicMLP(sizes).to(device)
            train_network(modelB, epochs=5)
            acc0_b = measure_accuracy(modelB)
            modelB.reset_activity_buffer()
            for x, y in list(train_loader)[:2]:
                modelB(x.to(device))
            modelB.apply_neuron_pruning(keep)          # baseline-Fallback
            optB = torch.optim.Adam(modelB.parameters(), lr=1e-3)
            for _ in range(6):
                modelB.train()
                modelB.reset_activity_buffer()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optB.zero_grad()
                    loss = F.cross_entropy(modelB(x), y)
                    loss.backward()
                    optB.step()
            acc_b = measure_accuracy(modelB)
            syn_b = modelB.active_synapses

            rows.append({
                "sizes": sizes, "seed": sd,
                "acc0": acc0_ml, "acc_ml": acc_ml, "acc_b": acc_b,
                "syn0": syn0, "syn_ml": syn_ml, "syn_b": syn_b,
                "keep": keep,
            })
            print(f"  [{li+1}/{len(test_layouts)}] {sizes} seed={sd} "
                  f"acc_vor={acc0_ml:.2f} | baseline acc={acc_b:.2f} "
                  f"(syn {syn_b:,}) | ML zero-shot acc={acc_ml:.2f} "
                  f"(syn {syn_ml:,})", flush=True)

    # Zusammenfassung + Plot
    print_table(rows)
    plot_zero_shot(rows)
    save_csv(rows)


def print_table(rows):
    print("\n" + "=" * 100)
    print("  ZERO-SHOT-AUSWERTUNG (Mittelwert ueber Seeds)")
    print("=" * 100)
    print(f"  {'Layout':<20}{'Acc voll':>9}{'Baseline':>10}{'ML zero':>9}"
          f"{'delta ML-Base':>14}")
    print("-" * 100)
    seen = {}
    for r in rows:
        seen.setdefault(tuple(r["sizes"]), []).append(r)
    for key, rs in seen.items():
        a0 = np.mean([r["acc0"] for r in rs])
        ab = np.mean([r["acc_b"] for r in rs])
        am = np.mean([r["acc_ml"] for r in rs])
        print(f"  {str(list(key)):<20}{a0:>8.2f}%{ab:>9.2f}%{am:>8.2f}%"
              f"{am - ab:>+13.2f}")
    print("=" * 100)


def plot_zero_shot(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        lbl = "ML (zero-shot)" if r["seed"] == LAYOUT_SEEDS[0] else None
        ax.scatter(r["syn_ml"], r["acc_ml"], color="#d62728", marker="o",
                   s=60, label=lbl if r["seed"] == LAYOUT_SEEDS[0] else None)
        ax.scatter(r["syn_b"], r["acc_b"], color="#1f77b4", marker="s",
                   s=60, label="Baseline" if r["seed"] == LAYOUT_SEEDS[0] else None)
        ax.scatter(r["syn0"], r["acc0"], color="#2ca02c", marker="^", s=50,
                   label="Voll (vor Prune)" if r["seed"] == LAYOUT_SEEDS[0] else None)
    ax.set_xlabel("Synapsen nach Pruning")
    ax.set_ylabel("Test-Accuracy (%)")
    ax.set_title("Zero-shot: trainiertes ML auf ungesehenen Layouts")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("zero_shot_result.png", dpi=150)
    print("  Plot gespeichert: zero_shot_result.png", flush=True)


def save_csv(rows):
    import csv
    with open("zero_shot_result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  CSV gespeichert: zero_shot_result.csv", flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
