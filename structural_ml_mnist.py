"""
Structural ML Controller on MNIST
==================================
Neue Idee (Platzhalter-Gerüst):

Das Hauptnetzwerk (ein MLP auf MNIST) wird weiterhin klassisch mit Backpropagation
trainiert. Zusätzlich gibt es einen EXTERNEN Machine-Learning-Algorithmus
(`StrukturController`), der die Netzwerkstruktur steuert:

  Input  : Korrelationsmatrix der Hidden-Neuronen-Aktivierungen
          + aktuelle Anzahl Neuronen/Synapsen
          + Ziel-Anzahl Neuronen/Synapsen
  Output : Entscheidung, WANN und WO das Netzwerk wachsen bzw. schrumpfen soll
  Ziel   : Nach X Epochen ist die Pruning-Phase beendet und das Netzwerk
           ist auf die Zielmenge (Neuronen, Synapsen) geschrumpft.

Dieses Programm liefert das GERÜST:
  * Korrelationsmatrix-Berechnung  -> fertig
  * Netzstruktur (Masks)           -> fertig (strukturiertes Pruning via Aktivitäts-Masks)
  * StrukturController             -> PLATZHALTER (TODO-Gerüst, noch KEIN ML-Algorithmus)
  * Fallback-Pruning               -> fertig, damit das Netz trotzdem messbar schrumpft
                                       (nur als Stand-in, bis der ML-Kontroller da ist)

Der ML-Algorithmus wird später in `StrukturController.decide()` implementiert.
Bis dahin wird das Ziel über das markierte Fallback-Pruning erreicht.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    """
    MLP (784 → h1 → h2 → h3 → 10) mit pro-Layer 'neuron_mask'.
    - Maskierte (0) Neuronen liefern keinen Beitrag mehr (strukturiertes Pruning).
    - 'active_count' pro Layer = Anzahl der nicht-maskierten Hidden-Neuronen.
    - 'active_synapses' = Anzahl der effektiv genutzten Gewichte.

    Korrelationsmatrix:
    - Meldet, manuell zu füllen, alle Aktivierungsvektoren der Hidden-Neuronen.
    - Aus diesem Puffer wird die Pearson-Korrelationsmatrix berechnet.
    """

    HIDDEN_SIZES = (256, 128, 64)

    def __init__(self):
        super().__init__()
        sizes = self.HIDDEN_SIZES
        self.fc1 = nn.Linear(28 * 28, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], sizes[2])
        self.fc4 = nn.Linear(sizes[2], 10)

        # Initial alle Neuronen aktiv
        self.neuron_masks = [
            torch.ones(s, dtype=torch.float32)
            for s in sizes
        ]
        # Puffer für Korrelationsmatrix
        self._act_buffer = []          # Liste von (Batch, tot_hidden) Aktivierungen
        self._corr = None              # (tot_hidden, tot_hidden)

    # ── Hidden-Größen ───────────────────────────────────────────────────
    @property
    def hidden_start_indices(self):
        s = self.HIDDEN_SIZES
        out = []
        acc = 0
        for i, sz in enumerate(s):
            out.append(acc)
            acc += sz
        return out

    @property
    def gap_tot_hidden(self):
        return int(sum(self.HIDDEN_SIZES))

    @property
    def neuron_counts(self):
        """Aktive Hidden-Neuronen pro Layer (strukturiert)."""
        return [int(m.sum().item()) for m in self.neuron_masks]

    @property
    def active_synapses(self):
        """Anzahl effektiv genutzter Gewichte (maskierte Einträge abgezogen)."""
        wp = [
            self.fc1.weight, self.fc2.weight, self.fc3.weight, self.fc4.weight
        ]
        sync = 0
        # fc1: Eingang (784) -> h1 (mask-Spalten in fc1)
        sync += wp[0].shape[1] * self.neuron_counts[0]
        # fc2: h1 -> h2
        sync += self.neuron_counts[0] * self.neuron_counts[1]
        # fc3: h2 -> h3
        sync += self.neuron_counts[1] * self.neuron_counts[2]
        # fc4: h3 -> 10
        sync += self.neuron_counts[2] * 10
        return sync

    # ── Forward mit Masken + Aktivierungsaufzeichnung ─────────────────
    def forward(self, x, record=True):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)) * self.neuron_masks[0].unsqueeze(0)
        h2 = F.relu(self.fc2(h1)) * self.neuron_masks[1].unsqueeze(0)
        h3 = F.relu(self.fc3(h2)) * self.neuron_masks[2].unsqueeze(0)
        out = self.fc4(h3)
        if record:
            self._act_buffer.append(
                torch.cat([h1.detach(), h2.view(h2.size(0), -1).detach(),
                           h3.view(h3.size(0), -1).detach()], dim=1)
            )
        return out

    def reset_activity_buffer(self):
        self._act_buffer = []

    # ── Korrelationsmatrix ───────────────────────────────────────────────
    def compute_correlation_matrix(self):
        """
        Berechnet die Pearson-Korrelationsmatrix über alle aktuell aufgezeichneten
        Aktivierungsvektoren. Ergebnis: (tot_hidden, tot_hidden) mit [-1, 1].
        NaN-Werte (konstante Neuronen) werden auf 0 gesetzt.
        """
        if len(self._act_buffer) == 0:
            self._corr = np.zeros((self.gap_tot_hidden, self.gap_tot_hidden))
            return self._corr
        acts = torch.cat(self._act_buffer, dim=0).cpu().numpy()  # (N, tot_hidden)
        # Downsampling für Performance
        if acts.shape[0] > 4096:
            idx = np.random.choice(acts.shape[0], 4096, replace=False)
            acts = acts[idx]
        acts_centered = acts - acts.mean(axis=0, keepdims=True)
        std = acts_centered.std(axis=0)
        std[std < 1e-8] = 1e-8
        acts_norm = acts_centered / std
        corr = (acts_norm.T @ acts_norm) / acts_norm.shape[0]
        corr = np.nan_to_num(corr, nan=0.0)
        self._corr = corr
        return corr

    # ── Struktur-Pruning (Masken setzen) ────────────────────────────────
    def apply_neuron_pruning(self, keep_per_layer):
        """
        Setzt neuron_masks so, dass pro Layer nur 'keep_per_layer[i]' Neuronen
        aktiv bleiben. Auswahl: kleinste mittlere Aktivierung zuerst entfernt.
        (Fallback; die Auswahl trifft später der ML-Kontroller.)
        """
        # Mittlere Aktivierung aus dem Puffer je Neuron (näherungsweise)
        acts = torch.cat(self._act_buffer, dim=0)
        starts = self.hidden_start_indices
        for i, sz in enumerate(self.HIDDEN_SIZES):
            keep = int(keep_per_layer[i])
            layer_acts = acts[:, starts[i]:starts[i] + sz].mean(dim=0)
            keep_idx = torch.argsort(layer_acts, descending=True)[:keep]
            new_mask = torch.zeros(sz, dtype=torch.float32)
            new_mask[keep_idx] = 1.0
            self.neuron_masks[i] = new_mask


# ═══════════════════════════════════════════════════════════════════════════
# 2) EXTERNER ML-KONTROLLER — PLATZHALTER-GERÜST (NOCH KEIN ML-Algorithmus)
# ═══════════════════════════════════════════════════════════════════════════
class StrukturController:
    """
    EXTERNER Machine-Learning-Algorithmus zur Steuerung der Netzwerkstruktur.

    ═══════════════════════════════════════════════════════════════════
    PLATZHALTER — hier wird der eigentliche ML-Algorithmus implementiert.
    ═══════════════════════════════════════════════════════════════════

    Idee (laut Vorgabe):
      * Das Hauptnetzwerk trainiert weiterhin mit Backpropagation.
      * Dieser Kontroller lernt (offline / separat), anhand der
        Korrelationsmatrix zu entscheiden, WANN und WO die Struktur
        wachsen bzw. schrumpfen soll.
      * Ziel: nach 'prune_epochs' Epochen ist das Netzwerk auf die
        Ziel-Anzahl Neuronen + Synapsen geschrumpft.

    Ein-/Ausgänge von decide():
      Input:
        corr            : np.ndarray (tot_hidden, tot_hidden) Korrelationsmatrix
        current_neurons : Liste von aktiven Hidden-Neuronen pro Layer
        current_synapses: int  aktive Synapsen
        target_neurons  : Liste von Ziel-Neuronen pro Layer
        target_synapses : int  Ziel-Synapsen
        epoch           : aktuelle Epoche
      Output (dict):
        action          : "prune" | "grow" | "none"
        keep_per_layer  : wie viele Hidden-Neuronen pro Layer aktiv bleiben
        reason          : Begründung/Log für Debug-Zwecke

    TODO-STEPS (sobald implementiert):
      1. Korrelationsmatrix zu einem Feature-Vektor verarbeiten
         (z.B. aggregieren: mean/std der Korrelation, Cluster der Neuronen).
      2. ML-Modell (z.B. kleines NN / Policy) als self.decision_model
         laden bzw. trainieren.
      3. Entscheidung aus self.decision_model(corr_features, counts, targets)
         ableiten: wann wachsen, wann schrumpfen, welche Neuronen.
      4. Lineares/adaptives Herunterfahren auf die Zielmenge bis prune_epochs.
    """

    def __init__(self, target_neurons, target_synapses, prune_epochs=10,
                 prune_interval=2):
        self.target_neurons = list(target_neurons)
        self.target_synapses = int(target_synapses)
        self.prune_epochs = prune_epochs
        self.prune_interval = prune_interval

        # ── PLATZHALTER: ML-Modell des Kontrollers (noch nicht trainiert) ──
        # self.decision_model = None  # TODO: hier das ML-Modell einbauen
        self.reason_history = []

    def decide(self, corr, current_neurons, current_synapses, epoch):
        """
        PLATZHALTER — entscheidet, ob / wie die Struktur sich ändert.
        Aktuell: lineares Fallback-Pruning bis zur Ziel-Neuronenzahl.
        Der eigentliche ML-Algorithmus ersetzt diesen Rumpf (TODO oben).
        """
        keep_per_layer = list(current_neurons)
        action = "none"

        if epoch >= 1 and epoch % self.prune_interval == 0:
            # Lineare Annäherung an die Ziel-Neuronenzahl
            progress = min(1.0, epoch / max(1, self.prune_epochs))
            new_counts = []
            for cur, tgt in zip(current_neurons, self.target_neurons):
                target_here = cur + (tgt - cur) * progress
                new_counts.append(int(round(target_here)))
            # Nicht unter die Zielmenge fallen bzw. nicht darüber hinaus wachsen
            new_counts = [
                int(min(max(n, t), c))
                for n, t, c in zip(new_counts, self.target_neurons, current_neurons)
            ]
            if sum(new_counts) < sum(current_neurons):
                keep_per_layer = new_counts
                action = "prune"

        result = {
            "action": action,
            "keep_per_layer": keep_per_layer,
            "current_neurons": list(current_neurons),
            "target_neurons": self.target_neurons,
            "corr_mean_abs": float(np.abs(corr).mean()) if corr is not None else 0.0,
            "epoch": epoch,
        }
        self.reason_history.append(result)
        return result

    def prune_finished(self, current_neurons, epoch):
        """True, wenn die Ziel-Neuronenzahl erreicht ist (Pruning-Phase abgeschlossen)."""
        return all(c == t for c, t in zip(current_neurons, self.target_neurons))


# ═══════════════════════════════════════════════════════════════════════════
# 3) Training mit struktur-Pruning über den Kontroller
# ═══════════════════════════════════════════════════════════════════════════
def train():
    model = MLPWithStruktur().to(device)
    target_neurons = [96, 48, 24]          # Ziel: 256→96, 128→48, 64→24
    target_synapses = model.active_synapses  # Ziel-Synapsen (hier: unverändert lassen)

    controller = StrukturController(
        target_neurons=target_neurons,
        target_synapses=target_synapses,
        prune_epochs=10,
        prune_interval=2,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 12
    print("\nStarte Training mit struktur-Controller (Platzhalter)...")
    print(f"  Start  Neuronen: {model.neuron_counts}  Synapsen: {model.active_synapses:,}")
    print(f"  Ziel   Neuronen: {target_neurons}  Synapsen: {target_synapses:,}\n")

    for ep in range(epochs):
        model.train()
        model.reset_activity_buffer()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # Kontroller befragen -> Struktur anpassen
        corr = model.compute_correlation_matrix()
        decision = controller.decide(
            corr, model.neuron_counts, model.active_synapses, ep + 1
        )
        if decision["action"] == "prune":
            model.apply_neuron_pruning(decision["keep_per_layer"])

        # Test-Accuracy
        model.eval()
        with torch.no_grad():
            correct, total_n = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x, record=False).argmax(1) == y).sum().item()
                total_n += y.size(0)
        acc = 100 * correct / total_n

        print(f"  ep {ep+1:2d} loss={total_loss/len(train_loader):.4f} "
              f"acc={acc:.2f}%  Neuronen={model.neuron_counts} "
              f"Syn={model.active_synapses:,}  corr={abs(corr).mean():.3f}",
              flush=True)

    print("\nFertig.")
    print(f"  Final Neuronen: {model.neuron_counts}  Synapsen: {model.active_synapses:,}")
    print(f"  Pruning-Phase abgeschlossen: "
          f"{controller.prune_finished(model.neuron_counts, epochs)}")


if __name__ == "__main__":
    train()
