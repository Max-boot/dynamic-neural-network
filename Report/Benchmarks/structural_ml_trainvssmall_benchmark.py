"""
Benchmark: "Trainieren & Prunen" vs. "Direkt klein trainieren"
==============================================================
Fragestellung (dein ESP32-Szenario):
  Ist es besser, ein VOLES Netz 784->256->128->64->10 zu trainieren,
  es physisch auf [96,48,24] zu kompaktieren und zu verfeinern -
  ODER direkt ein gleich grosses, schlankes Netz 784->96->48->24->10
  von Anfang an zu trainieren?

Am Ende haben beide Varianten die GLEICHE Architektur (81.264 Synapsen).
Der Benchmark zieht drei Konfigurationen:
  1) VOLL      : 784->256->128->64->10 (Referenz, grosse Architektur)
  2) PRUNE     : 784->256->128->64->10 trainieren -> physisch auf
                 [96,48,24] kompaktieren -> Fine-Tune (dein ESP32-Fall)
  3) SMALL     : direkt 784->96->48->24->10 trainieren (gleich gross)

Verglichen wird: Test-Accuracy, Synapsen/MACs, und (bei gleicher Groesse)
wer die bessere Accuracy erzielt.

Ausgabe:
  - Konsolen-Tabelle
  - Balkendiagramm (Accuracy)   RESULT_bild_trainvssmall_acc.png
  - Rohdaten (CSV)              RESULT_trainvssmall.csv
"""

import os, sys, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ── MNIST (robuster Pfad: nutzt das existierende /Code/mnist, sonst download) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_MNIST = r"D:\Dynamic Neural Networt (DNN)\Code\mnist"
if not os.path.isdir(_MNIST):
    _MNIST = os.path.join(_HERE, "mnist")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(_MNIST, train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(_MNIST, train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4096, shuffle=False)
print(f"MNIST unter: {_MNIST}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Kompaktierbares MLP
# ═══════════════════════════════════════════════════════════════════════
class MLPComp(nn.Module):
    """MLP mit dynamischen Schichtgroessen; forward ohne Masken."""
    def __init__(self, sizes):
        super().__init__()
        self.sizes = list(sizes)
        din = 28 * 28
        self.layers = nn.ModuleList()
        for s in sizes:
            self.layers.append(nn.Linear(din, s))
            din = s
        self.layers.append(nn.Linear(din, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, lay in enumerate(self.layers[:-1]):
            x = F.relu(lay(x))
        return self.layers[-1](x)

    def count_synapses(self):
        total = 0
        for lay in self.layers:
            total += lay.weight.shape[0] * lay.weight.shape[1]
        return int(total)


def neuron_select(W, keep):
    """Behaltene Neuronen einer Schicht nach mittlerer Aktivierung ueber
    einen Aktivierungs-Puffer (falls vorhanden), sonst Fan-in-Norm."""
    # Hier: Fan-in-Norm (reproduzierbar, vom Puffer unabhaengig)
    with torch.no_grad():
        return torch.argsort(W.norm(dim=1), descending=True)[:int(keep)]


def compact_physical(src, keep_counts):
    """Kopiert src-MLP in ein NEUES, physisch auf keep_counts geschrumpftes MLP.
    src: MLPComp mit len(src.layers)-1 == 3 Hidden-Layer."""
    with torch.no_grad():
        keep0 = neuron_select(src.layers[0].weight, keep_counts[0])  # aus 256
        keep1 = neuron_select(src.layers[1].weight, keep_counts[1])  # aus 128
        keep2 = neuron_select(src.layers[2].weight, keep_counts[2])  # aus 64

    nc = MLPComp(keep_counts)
    L, C = src.layers, nc.layers
    with torch.no_grad():
        C[0].weight.copy_(L[0].weight[keep0])           # (keep0, 784)
        C[0].bias.copy_(L[0].bias[keep0])
        C[1].weight.copy_(L[1].weight[keep1][:, keep0])  # (keep1, keep0)
        C[1].bias.copy_(L[1].bias[keep1])
        C[2].weight.copy_(L[2].weight[keep2][:, keep1])  # (keep2, keep1)
        C[2].bias.copy_(L[2].bias[keep2])
        C[3].weight.copy_(L[3].weight[:, keep2])         # (10, keep2)
        C[3].bias.copy_(L[3].bias)
    return nc


def measure_accuracy(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


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


# ═══════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════
FULL   = [256, 128, 64]
PRUNE  = [96, 48, 24]   # Zielgroesse nach Kompaktierung
WARM   = 6              # Epochen vor dem Kompaktieren (gross)
FINETUNE = 6            # Epochen nach dem Kompaktieren
TOTAL  = WARM + FINETUNE  # faires Training samt FUER das Vergleichsnetz

def run_benchmark(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) VOLLES Netz als Referenz
    model_full = MLPComp(FULL).to(device)
    train_epochs(model_full, TOTAL)
    acc_full = measure_accuracy(model_full)

    # 2) PRUNE: voll trainieren -> kompaktieren -> fine-tune
    torch.manual_seed(seed); np.random.seed(seed)
    model_prune = MLPComp(FULL).to(device)
    train_epochs(model_prune, WARM)
    model_prune_c = compact_physical(model_prune, PRUNE).to(device)
    train_epochs(model_prune_c, FINETUNE)
    acc_prune = measure_accuracy(model_prune_c)

    # 3) SMALL: direkt klein trainieren (gleich viele Epochen gesamt)
    torch.manual_seed(seed); np.random.seed(seed)
    model_small = MLPComp(PRUNE).to(device)
    train_epochs(model_small, TOTAL)
    acc_small = measure_accuracy(model_small)

    return {
        "seed": seed,
        "acc_full": acc_full,
        "acc_prune": acc_prune,
        "acc_small": acc_small,
        "syn_full": model_full.count_synapses(),
        "syn_prune": model_prune_c.count_synapses(),
        "syn_small": model_small.count_synapses(),
    }


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [42, 2024]
    rows = [run_benchmark(s) for s in seeds]

    # Aggregieren (Mittelwert)
    mean = lambda k: np.mean([r[k] for r in rows])
    a_full, a_prune, a_small = mean("acc_full"), mean("acc_prune"), mean("acc_small")
    sy_full = int(mean("syn_full"))
    sy_prune = int(mean("syn_prune"))
    sy_small = int(mean("syn_small"))

    print("\n" + "=" * 78)
    print("BENCHMARK: Trainieren&Prunen vs. Direkt-klein-trainieren (MNIST)")
    print("=" * 78)
    print(f"  Seeds: {seeds}  |  (Mittelwerte ueber Seeds)")
    print("-" * 78)
    print(f"  {'Variante':<26}{'Acc (%):':>10}{'Syn/MACs':>12}")
    print("-" * 78)
    print(f"  {'VOLL  784->'+str(FULL):<26}{a_full:>10.2f}{sy_full:>12,}")
    print(f"  {'PRUNE ->'+str(PRUNE):<26}{a_prune:>10.2f}{sy_prune:>12,}")
    print(f"  {'SMALL '+str(PRUNE):<26}{a_small:>10.2f}{sy_small:>12,}")
    print("-" * 78)
    print(f"  Synapsen: VOLL={sy_full:,} | PRUNE & SMALL beiden={sy_prune:,}")
    print(f"  -> PRUNE und SMALL sind identisch gross (physisch kompaktiert).")
    print(f"  PRUNE - VOLL : {a_prune - a_full:+.2f} pp")
    print(f"  SMALL - VOLL : {a_small - a_full:+.2f} pp")
    print(f"  PRUNE - SMALL: {a_prune - a_small:+.2f} pp  (PRUNE besser -> "
          f"+; SMALL besser -> -)")
    print("=" * 78)

    # Plot
    labels = [f"VOLL\n{[FULL]}", f"PRUNE\n->{[PRUNE]}", f"SMALL\n{[PRUNE]}"]
    vals = [a_full, a_prune, a_small]
    colors = ["#888888", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%",
                ha="center", fontweight="bold")
    ax.axhline(a_full, color="#888", ls="--", alpha=0.6, label="VOLL (Referenz)")
    ax.set_ylabel("Test-Accuracy (%)")
    ax.set_title("Trainieren & physisch Prunen vs. direkt klein trainieren (gleiche Groesse)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = os.path.join(_HERE, "RESULT_bild_trainvssmall_acc.png")
    fig.savefig(png, dpi=150)
    print(f"  Plot gespeichert: {png}")

    # CSV
    csvp = os.path.join(_HERE, "RESULT_trainvssmall.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "acc_full", "acc_prune", "acc_small",
                    "syn_full", "syn_prune", "syn_small"])
        for r in rows:
            w.writerow([r[k] for k in
                        ("seed", "acc_full", "acc_prune", "acc_small",
                         "syn_full", "syn_prune", "syn_small")])
    print(f"  CSV gespeichert: {csvp}")


if __name__ == "__main__":
    main()
