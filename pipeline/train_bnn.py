"""
Training Stufe 4+5: BNN (MC-Dropout) + Bounding-Box-Head auf Scene-Crops.

Finale Konfiguration (nach Ablation, siehe README_PIPELINE):
  - 'Kapazitaet + mehr Daten': c1=40, c2=80, hid=192 (0.79M Params),
    mehr Positiv-Crops (window x10, tile x5, n_cluttered=20000) und
    70 Epochen. Erhoht val_acc 0.549 -> 0.573 (digit_acc 0.460).
  - Verworfen: MNIST-Backbone-Pretrain (0.534) und geometrische
    Rotation/Zoom-Augmentation (0.504) - beide senkten die Acc auf der
    verrauschten Scene-Verteilung; im Notebook als Ablation berichtet.

Speichert Modell, Verlaeufe, Val-Acc je Epoche (inkl. Ziffern-Fehlerrate).
"""
import errno
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnn import BNN, train_bnn
from bnn_data import build_bnn_datasets

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

C1, C2, HID = 40, 80, 192


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    tr_ds, te_ds = build_bnn_datasets(seed=42, use_cluttered=True,
                                      max_pos_window=10, max_pos_tile=5,
                                      neg_per_img_train=5, neg_per_img_test=4,
                                      n_cluttered=20000)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"Train ", len(tr_ds), "Test", len(te_ds))

    torch.manual_seed(42)
    model = BNN(n_class=11, dropout=0.3, box_head=True, c1=C1, c2=C2, hid=HID)
    print("BNN-Parameter:", sum(p.numel() for p in model.parameters()))
    losses, accs, daccs = train_bnn(model, tr_ld, te_ld, epochs=70, lr=1e-3,
                                    device=dev, box_weight=3.0)

    torch.save({"model": model.state_dict(),
                "losses": losses, "val_accs": accs, "val_digit_accs": daccs,
                "c1": C1, "c2": C2, "hid": HID},
               os.path.join(MODELS, "bnn_mc_box.pt"))
    print(f"Final val_acc={accs[-1]:.4f} digit_acc={daccs[-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(losses)
    ax[0].set(title="BNN Training-Loss (CE+Box)", xlabel="Epoche")
    ax[0].grid(alpha=0.3)
    ax[1].plot(accs, label="val_acc (alle)")
    ax[1].plot(daccs, label="val_acc (Ziffern 0-9)")
    ax[1].set(title="BNN Val-Accuracy", xlabel="Epoche")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "bnn_training.png"), dpi=150)

    import csv
    with open(os.path.join(RESULTS, "bnn_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["final_val_acc", f"{accs[-1]:.4f}"])
        w.writerow(["final_digit_acc", f"{daccs[-1]:.4f}"])
        w.writerow(["n_params", sum(p.numel() for p in model.parameters())])
    print("Fertig.")


if __name__ == "__main__":
    main()