"""
Training Stufe 4+5: BNN (MC-Dropout) + Bounding-Box-Head auf Scene-Crops.

Speichert Modell, Verlaeufe, Val-Acc je Epoche (inkl. Hintergrund-Klasse).
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


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    tr_ds, te_ds = build_bnn_datasets(seed=42, use_cluttered=True,
                                      max_pos_window=8, max_pos_tile=4,
                                      neg_per_img_train=4, neg_per_img_test=4,
                                      n_cluttered=16000)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"Train ", len(tr_ds), "Test", len(te_ds))

    torch.manual_seed(42)
    model = BNN(n_class=11, dropout=0.3, box_head=True, c1=32, c2=64, hid=128)
    losses, accs = train_bnn(model, tr_ld, te_ld, epochs=45, lr=1e-3,
                             device=dev, box_weight=3.0)

    torch.save({"model": model.state_dict(),
                "losses": losses, "val_accs": accs},
               os.path.join(MODELS, "bnn_mc_box.pt"))
    print(f"Final val_acc={accs[-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(losses)
    ax[0].set(title="BNN Training-Loss (CE+Box)", xlabel="Epoche")
    ax[0].grid(alpha=0.3)
    ax[1].plot(accs)
    ax[1].set(title="BNN Val-Accuracy", xlabel="Epoche")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "bnn_training.png"), dpi=150)

    import csv
    with open(os.path.join(RESULTS, "bnn_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["final_val_acc", f"{accs[-1]:.4f}"])
        w.writerow(["n_params", sum(p.numel() for p in model.parameters())])
    print("Fertig.")


if __name__ == "__main__":
    main()