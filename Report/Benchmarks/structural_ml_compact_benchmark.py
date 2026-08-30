"""
Laufzeit-Benchmark: Masken-Pruning vs. physisches Kompaktieren
===============================================================
Beweis der These: Masken-Pruning (Aktivierungen nullen) bringt praktisch KEINEN
Laufzeitgewinn, weil die Gewichtsmatrizen voll bleiben und jede Matrixmultiplikation
weiter ueber alle Kanäle laeuft.

Physisches Kompaktieren schrumpft die Matrizen WIRKLICH (W -> W[keep_idx]) und
reduziert damit die Zahl der ausgefuehrten Synapsen-Multiplikationen.

Misst die Forward-Zeit einer festen Batch-Groesse auf GPU in 3 Zustaenden:
    full     : volles Netz, keine Masken
    masked   : nach Masken-Pruning (Matrizen VOLL, Aktivierungen genullt)
    compact  : nach physischem Kompaktieren (Matrizen GESCHRUMPFT)
Accuracy muss bei masked == compact sein (identische Mathematik nach Kompaktieren).
"""

import time
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

# ── MNIST ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./mnist", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1024, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════
# Netz mit uns (Aktivierungs-Masken) UND kompakter Variante
# ═══════════════════════════════════════════════════════════════════════
class MLPComp(nn.Module):
    def __init__(self, sizes=(256, 128, 64)):
        super().__init__()
        self.sizes = list(sizes)
        self.fc1 = nn.Linear(28 * 28, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], sizes[2])
        self.fc4 = nn.Linear(sizes[2], 10)

    # Forward mit Maske: Aktivierungen nullen, Matrizen bleiben voll
    def forward_masked(self, x, masks):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)) * masks[0]
        h2 = F.relu(self.fc2(h1)) * masks[1]
        h3 = F.relu(self.fc3(h2)) * masks[2]
        return self.fc4(h3)

    # Forward ohne Maske (volles oder kompaktes Netz)
    def forward_full(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc4(h3)

    def count_synapses(self):
        w = [self.fc1.weight, self.fc2.weight, self.fc3.weight, self.fc4.weight]
        return (w[0].shape[1] * w[0].shape[0]
                + w[1].shape[0] * w[1].shape[1]
                + w[2].shape[0] * w[2].shape[1]
                + w[3].shape[0] * w[3].shape[1])


def neuron_select(W, keep):
    """Waehlt 'keep' Neuronen einer Schicht anhand der L2-Norm ihrer Zeilen.
    W ist die Gewichtsmatrix der Schicht der Form (out, in); Zeilen = Neuronen.
    (Fan-in-Norm ist fuer den Laufzeit-Beweis egal; Mittel-Aktivierung gleichwertig.)
    """
    with torch.no_grad():
        return torch.argsort(W.norm(dim=1), descending=True)[:int(keep)]


def compact_physical(model, keep_counts):
    """Gibt ein NEUES MLPComp zurueck, dessen Matrizen WIRKLICH auf die aktiven
    Neuronen geschrumpft sind. Die Neuron-Auswahl je Schicht ist unabhaengig.
    """
    with torch.no_grad():
        # je Schicht eigene Neuronenliste (0..sz-1)
        keep0 = neuron_select(model.fc1.weight, keep_counts[0])   # aus 256
        keep1 = neuron_select(model.fc2.weight, keep_counts[1])   # aus 128
        keep2 = neuron_select(model.fc3.weight, keep_counts[2])   # aus 64

    nc = MLPComp(keep_counts)
    with torch.no_grad():
        # fc1.weight (256,784) -> (keep0,784)
        nc.fc1.weight.copy_(model.fc1.weight[keep0])
        nc.fc1.bias.copy_(model.fc1.bias[keep0])
        # fc2.weight (128,256) -> (keep1, keep0)
        nc.fc2.weight.copy_(model.fc2.weight[keep1][:, keep0])
        nc.fc2.bias.copy_(model.fc2.bias[keep1])
        # fc3.weight (64,128) -> (keep2, keep1)
        nc.fc3.weight.copy_(model.fc3.weight[keep2][:, keep1])
        nc.fc3.bias.copy_(model.fc3.bias[keep2])
        # fc4.weight (10,64) -> (10, keep2)
        nc.fc4.weight.copy_(model.fc4.weight[:, keep2])
        nc.fc4.bias.copy_(model.fc4.bias)
    return nc


# ═══════════════════════════════════════════════════════════════════════
# Training (kurz, um ein sinnvolles, partiell gelerntes Netz zu haben)
# ═══════════════════════════════════════════════════════════════════════
def model_device(model):
    return next(model.parameters()).device


def train_short(model, epochs=4):
    dev = model_device(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model.forward_full(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()


def accuracy(model, forward_fn):
    dev = model_device(model)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            correct += (forward_fn(model, x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# ═══════════════════════════════════════════════════════════════════════
# Laufzeit-Messung (Wall-Time ueber viele Forward-Passes, Warmup)
# ═══════════════════════════════════════════════════════════════════════
def bench_forward(model, forward_fn, x_batch, reps=200):
    allow_cuda = model_device(model).type == "cuda"
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            forward_fn(model, x_batch)          # Warmup (Kernel-Loading)
        if allow_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            forward_fn(model, x_batch)
        if allow_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / reps * 1000  # ms / Forward


def run_benchmark(dev_str):
    """Misst LaLaufzeit auf einem Geraet ('cuda' oder 'cpu').
    Liefert ein Dict mit allen Kennzahlen und gibt eine Tabelle aus."""
    dev = torch.device(dev_str)
    print(f"\n=== [{dev_str.upper()}] Training (kurz) ===", flush=True)
    torch.manual_seed(42); np.random.seed(42)
    model = MLPComp().to(dev)
    train_short(model, epochs=4)
    acc_full = accuracy(model, lambda m, x: m.forward_full(x))

    # feste Batch-Groesse fuer faire Messung
    xb, _ = next(iter(test_loader))
    xb = xb.to(dev)

    sizes = model.sizes
    keep_counts = [96, 48, 24]              # Prune-Ziel wie in v2

    # ── 1) Masken-Pruning ─────────────────────────────────────────────
    #   Aktivierungen der entfernten Neuronen nullen (Matrizen bleiben VOLL)
    masks = []
    with torch.no_grad():
        # aktive Indizes je Schicht (identisch zu compact_physical)
        keep0 = neuron_select(model.fc1.weight, keep_counts[0])   # aus 256
        keep1 = neuron_select(model.fc2.weight, keep_counts[1])   # aus 128
        keep2 = neuron_select(model.fc3.weight, keep_counts[2])   # aus 64
        m0 = torch.zeros(sizes[0], device=dev); m0[keep0] = 1.0
        m1 = torch.zeros(sizes[1], device=dev); m1[keep1] = 1.0
        m2 = torch.zeros(sizes[2], device=dev); m2[keep2] = 1.0
        masks = [m0.view(1, -1), m1.view(1, -1), m2.view(1, -1)]

    def forward_masked_wrap(m, x):
        return m.forward_masked(x, masks)

    acc_masked = accuracy(model, forward_masked_wrap)

    reps = 500 if dev_str == "cpu" else 200
    t_full    = bench_forward(model, lambda m, x: m.forward_full(x), xb, reps)
    t_masked  = bench_forward(model, forward_masked_wrap, xb, reps)

    # ── 2) Physisches Kompaktieren ────────────────────────────────────
    model_c = compact_physical(model, keep_counts).to(dev)
    acc_compact = accuracy(model_c, lambda m, x: m.forward_full(x))
    t_compact = bench_forward(model_c, lambda m, x: m.forward_full(x), xb, reps)

    src = model.count_synapses()
    dst = model_c.count_synapses()
    g_m = t_full / t_masked
    g_c = t_full / t_compact

    print("\n" + "=" * 74)
    print(f"LAUFZEIT-BENCHMARK  (batch={xb.size(0)}, {dev_str.upper()})")
    print("=" * 74)
    print(f"  volles Netz      : {t_full:8.4f} ms/Fwd   acc={acc_full:6.2f}%   "
          f"Syn={src:,}")
    print(f"  Masken-Pruning   : {t_masked:8.4f} ms/Fwd   acc={acc_masked:6.2f}%   "
          f"Syn={src:,}   (Matrizen VOLL)")
    print(f"  Physisch kompakt : {t_compact:8.4f} ms/Fwd   acc={acc_compact:6.2f}%   "
          f"Syn={dst:,}   (Matrizen geschrumpft)")
    print("-" * 74)
    print(f"  Speedup Masken-Pruning : {g_m:5.2f}x   (Erwartung ~1.0x)")
    print(f"  Speedup Kompaktieren   : {g_c:5.2f}x")
    print(f"  Synapsen-Reduktion     : {100 * (1 - dst / src):5.1f}%  "
          f"({src:,} -> {dst:,})")
    assert abs(acc_masked - acc_compact) < 1e-6, \
        "Masked und compact muessen identische Accuracy haben!"
    print("  (masked == compact bestaetigt: identische Mathematik)")
    print("=" * 74)

    return dict(device=dev_str, batch=xb.size(0), reps=reps,
                acc_full=acc_full, acc_masked=acc_masked, acc_compact=acc_compact,
                t_full=t_full, t_masked=t_masked, t_compact=t_compact,
                speedup_masked=g_m, speedup_compact=g_c,
                syn_full=src, syn_compact=dst, syn_reduction=100 * (1 - dst / src))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Masken- vs. Kompaktier-Benchmark (GPU/CPU)")
    p.add_argument("--devices", nargs="+", default=["cuda", "cpu"],
                   choices=["cuda", "cpu"], help="Zu messende Geraete")
    p.add_argument("--csv", default=None,
                   help="Optionaler Pfad zum Schreiben der Ergebnisse (CSV)")
    args = p.parse_args(argv)

    results = []
    for d in args.devices:
        if d == "cuda" and not torch.cuda.is_available():
            print("CUDA nicht verfuegbar, ueberspringe 'cuda'.")
            continue
        results.append(run_benchmark(d))

    if args.csv and results:
        import csv
        keys = list(results[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nErgebnisse geschrieben nach: {args.csv}")


if __name__ == "__main__":
    main()
