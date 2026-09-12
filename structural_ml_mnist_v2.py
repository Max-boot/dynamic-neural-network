"""
Structural ML Controller on MNIST — Version 2 (implementiert)
==============================================================
Verwendet die "Neue Idee": Ein externer Machine-Learning-Algorithmus steuert,
WANN und WO das (weiterhin mit Backpropagation trainierte) Hauptnetzwerk
geschrumpft wird — anhand der Korrelationsmatrix.

  Komponente A (Gate / "WANN"):
    - Reagiert auf die Accuracy: Wenn ein Pruning die Test-Accuracy zu stark
      einbrechen lässt, wird im nächsten Intervall NICHT gepruned (Safeguard),
      damit sich das Netz erst erholen kann.

  Komponente B (Selector / "WO"): der eigentliche Lern-Teil
    - Ein kleines MLP (NeuronScorer) wird trainiert, um pro Neuron dessen
      Wichtigkeit vorherzusagen.
    - Features pro Neuron (aus Korrelationsmatrix + Aktivierung + Klasse):
        f0: mittlere Aktivierung
        f1: Partizipationsrate (Anteil Samples, in denen Neuron feuert)
        f2: mittleres |korrelation| innerhalb der Schicht  (Redundanz)
        f3: max |korrelation| innerhalb der Schicht        (Redundanz)
        f4: Klassen-Spezifität (z-Wert der klassenbedingten Aktivierung)
            -> schützt Ziffern-spezifische Neuronen (deine Idee #1)
        f5: Schichtindex (normalisiert)
    - Label = Gradient-Salienz (|a * grad_L|) = wie stark ein Neuron zur
      Vorhersage beiträgt. (OBD-artige Wichtigkeit, eine echte Ground-Truth.)
    - Der trainierte Scorer entfernt gezielt die NIEDRIG-wichtigen Neuronen
      und schützt die hohen (z.B. Ziffern-spezifische).

Am Ende werden zwei Konfigurationen gefahren und verglichen:
  BASELINE  : Fallback-Pruning (Entfernt nach mittlerer Aktivierung)
  ML        : trainierter NeuronScorer (Entfernt nach ML-Wichtigkeit)

Verglichen wird: Accuracy vor/nach dem Pruning + Neuronen/Synapsen-Zahl.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1024, shuffle=False)
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1) Netzwerk mit Aktivitäts-Masks + Korrelationsmatrix
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

        self.neuron_masks = [torch.ones(s, dtype=torch.float32) for s in sizes]
        self._act_buffer = []
        self._corr = None
        # klassenbedingte Aktivierungs-Summen (für Spezifität)
        self.class_act_sum = torch.zeros((10, self.gap_tot_hidden))
        self.class_act_count = torch.zeros(10)

    # ── Hidden-Infrastruktur ─────────────────────────────────────────────
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

    # ── Forward ──────────────────────────────────────────────────────────
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
        """Gibt zusätzlich die verdeckten Aktivierungsvektoren zurück."""
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)) * self.neuron_masks[0].unsqueeze(0)
        h2 = F.relu(self.fc2(h1)) * self.neuron_masks[1].unsqueeze(0)
        h3 = F.relu(self.fc3(h2)) * self.neuron_masks[2].unsqueeze(0)
        return self.fc4(h3), h1, h2, h3

    def update_class_stats(self, x, y):
        """Akkumuliert klassenbedingte Aktivierungs-Summen (für Spezifität)."""
        with torch.no_grad():
            _, h1, h2, h3 = self.forward_repr(x)
            ha = torch.cat([h1, h2, h3], dim=1).detach().cpu()  # (B, tot)
            yc = y.cpu()
            for d in range(10):
                m = yc == d
                if m.any():
                    self.class_act_sum[d] += ha[m].sum(dim=0)
                    self.class_act_count[d] += m.sum()

    def class_specificity(self):
        """
        z-Wert der klassenbedingten Aktivierung je Neuron (pro Schicht-normiert).
        Hoher Wert = Neuron 'feuert' stark für genau eine Ziffer.
        """
        spec = torch.zeros(self.gap_tot_hidden)
        for i, sz in enumerate(self.HIDDEN_SIZES):
            lo = self.hidden_start_indices[i]
            hi = lo + sz
            counts = self.class_act_count.clone()
            counts[counts == 0] = 1.0
            cond_mean = (self.class_act_sum / counts.view(-1, 1))[:, lo:hi]  # (10, sz)
            gmean = cond_mean.mean(dim=0)            # (sz,)
            gstd = cond_mean.std(dim=0) + 1e-8
            dev = (cond_mean - gmean) / gstd
            spec[lo:hi] = dev.abs().max(dim=0).values
        return spec

    # ── Korrelationsmatrix ────────────────────────────────────────────────
    def compute_correlation_matrix(self):
        if len(self._act_buffer) == 0:
            self._corr = np.zeros((self.gap_tot_hidden, self.gap_tot_hidden))
            return self._corr
        acts = torch.cat(self._act_buffer, dim=0).cpu().numpy()
        if acts.shape[0] > 4096:
            acts = acts[np.random.choice(acts.shape[0], 4096, replace=False)]
        a = acts - acts.mean(axis=0, keepdims=True)
        sd = a.std(axis=0)
        sd[sd < 1e-8] = 1e-8
        corr = np.nan_to_num((a.T @ a) / a.shape[0])
        self._corr = corr
        return corr

    # ── Struktur-Pruning ──────────────────────────────────────────────────
    def apply_neuron_pruning(self, keep_per_layer, scores_all=None):
        """
        Setzt neuron_masks so, dass pro Schicht 'keep[layer]' Neuronen aktiv bleiben.
        - scores_all=None  -> Fallback: höchste mittlere Aktivierung behalten
        - scores_all!=None -> ML-Pruning: höchste ML-Wichtigkeit behalten
        """
        acts = torch.cat(self._act_buffer, dim=0)
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
            self.neuron_masks[i] = new_mask


# ═══════════════════════════════════════════════════════════════════════════
# 2) Feature-Extraktion + Gradient-Wichtigkeit (Ground-Truth für den Scorer)
# ═══════════════════════════════════════════════════════════════════════════
def compute_batch_importance(model, x, y):
    """
    OBD-artige Salienz je Neuron: mean |a * grad_L(a)| über den Batch.
    Liefert einen Vektor der Länge tot_hidden.
    """
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
    """
    Baut für alle aktuell aktiven Neuronen einen Feature-Vektor (n_feat).
    korrelation: (tot, tot). Nutzt Aktivierungs-Mittel + Spezifität.
    """
    n = model.gap_tot_hidden
    acts = torch.cat(model._act_buffer, dim=0) if model._act_buffer else \
        torch.zeros(1, n)
    act_mean = acts.mean(dim=0)                       # (tot,)
    part_rate = (acts > 0).float().mean(dim=0)         # (tot,)
    spec = model.class_specificity()                   # (tot,)
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
                      spec, layer_idx], axis=1)  # (n, 6)
    return feats


def build_selector_dataset(model, loader, n_batches):
    """
    Sammelt pro Batch die (Feature, Importance)-Paare für alle Neuronen.
    Liefert (X, y) zum Trainieren des NeuronScorer.
    """
    feats_list, imp_list = [], []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        model.update_class_stats(x, y)
        imp = compute_batch_importance(model, x, y)
        model.reset_activity_buffer()
        model(x.to(device))
        corr = model.compute_correlation_matrix()
        f = extract_features(model, corr)
        feats_list.append(f)
        imp_list.append(imp.numpy())
    X = np.concatenate(feats_list, axis=0)
    starts = model.hidden_start_indices
    sizes = model.HIDDEN_SIZES
    y_all = np.concatenate(imp_list, axis=0)
    for i, sz in enumerate(sizes):
        lo, hi = starts[i], starts[i] + sz
        col = y_all[:, lo:hi]
        col = col / (col.max() + 1e-8)
        y_all[:, lo:hi] = col
    return X, y_all


# ═══════════════════════════════════════════════════════════════════════════

class NeuronScorer(nn.Module):
    """Kleines MLP: Features -> Wichtigkeit (pro Neuron)."""
    def __init__(self, n_feat=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StrukturControllerML:
    """
    Externer ML-Kontroller mit trainiertem NeuronScorer.

    decide():
      - WANN: pruned nur alle 'prune_interval' Epochen und nur, wenn das
        letzte Pruning die Accuracy nicht zu stark (über 'acc_tolerance' %)
        gegenüber dem besten erreichten Wert einbrechen ließ (Safeguard ->
        das Netz darf sich erholen).
      - WO  : NeuronScorer bewertet jedes Neuron; entfernt die niedrig-
        wichtigen, behält die hoch-wichtigen (schützt Ziffern-spezifische).
    """

    def __init__(self, target_neurons, prune_epochs=10, prune_interval=2,
                 acc_tolerance=5.0):
        self.target_neurons = list(target_neurons)
        self.prune_epochs = prune_epochs
        self.prune_interval = prune_interval
        self.acc_tolerance = acc_tolerance
        self.scorer = NeuronScorer(n_feat=6).to(device)
        self.trained = False
        self.history = []
        self.best_acc = 0.0
        self.acc_after_last_prune = None

    def train_scorer_on_data(self, X, Y, epochs=300):
        """Trainiert den NeuronScorer auf bereits extrahierten (Features, Goals)."""
        Xt = torch.from_numpy(X).float().to(device)
        Yt = torch.from_numpy(Y).float().to(device)
        w = (Yt + 1e-3)  # unwichtige Neuronen leicht untergewichten
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

        # Best-Accuracy verfolgen
        self.best_acc = max(self.best_acc, last_acc)

        do_prune = epoch >= 1 and epoch % self.prune_interval == 0
        if do_prune:
            # Safeguard: pausieren, wenn das letzte Pruning zu sehr geschadet hat
            if (self.acc_after_last_prune is not None
                    and self.acc_after_last_prune < self.best_acc - self.acc_tolerance):
                action = "skip (recover)"
            else:
                progress = min(1.0, epoch / max(1, self.prune_epochs))
                new_counts = []
                for cur, tgt in zip(current_neurons, self.target_neurons):
                    target_here = cur + (tgt - cur) * progress
                    new_counts.append(int(round(target_here)))
                new_counts = [int(min(max(n, t), c)) for n, t, c in
                              zip(new_counts, self.target_neurons, current_neurons)]
                if sum(new_counts) < sum(current_neurons):
                    keep_per_layer = new_counts
                    action = "prune"

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
# 4) Training + Vergleich
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


def run_config(use_ml, seed, epochs=12, warmup_epochs=2, prune_epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPWithStruktur().to(device)
    target_neurons = [96, 48, 24]
    syn_start = model.active_synapses
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n{'='*64}")
    print(f"  {'ML KONTROLLER' if use_ml else 'BASELINE (Fallback)'}")
    print(f"{'='*64}", flush=True)

    # ── Warmup: volles Netz trainieren ───────────────────────────────────
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
    print(f"  [Warmup] Accuracy (volles Netz): {acc_start:.2f}%  "
          f"Neuronen={model.neuron_counts}  Syn={model.active_synapses:,}")

    controller = StrukturControllerML(target_neurons=target_neurons,
                                      prune_epochs=prune_epochs,
                                      prune_interval=2)

    # ── Scorer-Training im ML-Fall ───────────────────────────────────────
    if use_ml:
        model.train()
        sampler = iter(train_loader)
        all_feats = []
        all_imp = []
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
        X_all = np.concatenate(all_feats, axis=0)   # (n*6, 6)
        Y_all = np.concatenate(all_imp, axis=0)     # (n*6,)
        # Wichtigkeit je Schicht normalisieren (für faire Balance)
        starts = model.hidden_start_indices
        for i, sz in enumerate(model.HIDDEN_SIZES):
            lo, hi = starts[i], starts[i] + sz
            Y_all[lo:hi] = Y_all[lo:hi] / (Y_all[lo:hi].max() + 1e-8)
        controller.train_scorer_on_data(X_all, Y_all)
        print(f"  [Scorer] trainiert auf {Y_all.shape[0]} Neuronen-Samples")

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
        print(f"  ep {ep+1:2d} acc={acc:.2f}%  Neuronen={model.neuron_counts} "
              f"Syn={model.active_synapses:,}  action={decision['action']:>15}",
              flush=True)

    acc_end = accs[-1]
    print(f"  [Ende] Accuracy={acc_end:.2f}%  Neuronen={model.neuron_counts}  "
          f"Syn={model.active_synapses:,}")
    return {
        "acc_start": acc_start,
        "acc_end": acc_end,
        "neurons_start": list(MLPWithStruktur.HIDDEN_SIZES),
        "neurons_end": model.neuron_counts,
        "syn_start": syn_start,
        "syn_end": model.active_synapses,
    }


if __name__ == "__main__":
    baseline = run_config(use_ml=False, seed=42)
    ml_ctrl = run_config(use_ml=True, seed=42)

    print(f"\n{'='*80}")
    print("VERGLEICH — Baseline vs. ML-Kontroller")
    print(f"{'='*80}")
    print(f"  {'Metrik':<28}{'Baseline':>18}{'ML-Kontroller':>18}")
    print("-" * 66)
    print(f"  {'Accuracy vor Pruning':<28}{baseline['acc_start']:>17.2f}%"
          f"{ml_ctrl['acc_start']:>17.2f}%")
    print(f"  {'Accuracy nach Pruning':<28}{baseline['acc_end']:>17.2f}%"
          f"{ml_ctrl['acc_end']:>17.2f}%")
    b_delta = baseline['acc_end'] - baseline['acc_start']
    m_delta = ml_ctrl['acc_end'] - ml_ctrl['acc_start']
    print(f"  {'Delta Accuracy':<28}{b_delta:>+17.2f}%{m_delta:>+17.2f}%")
    print(f"  {'Neuronen Start':<28}{str(baseline['neurons_start']):>18}"
          f"{str(ml_ctrl['neurons_start']):>18}")
    print(f"  {'Neuronen Ende':<28}{str(baseline['neurons_end']):>18}"
          f"{str(ml_ctrl['neurons_end']):>18}")
    print(f"  {'Synapsen Start':<28}{baseline['syn_start']:>17,}"
          f"{ml_ctrl['syn_start']:>17,}")
    print(f"  {'Synapsen Ende':<28}{baseline['syn_end']:>17,}"
          f"{ml_ctrl['syn_end']:>17,}")
    print("=" * 80)
