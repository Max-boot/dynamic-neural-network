"""
Trainings-Skript Stufe 1+2 (Conv+ANFIS Saliency) auf dem Scene-Split.

Trainiert das gemeinsame ConvANFISSaliency-Modell end-to-end (BCE),
speichert Modell + Metriken + Saliency-Visualisierungen.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, boxes_to_tiles
from stage12 import (ConvANFISSaliency, train_stage12, eval_stage12,
                     pos_weight_from)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    tr = load_scene_split("train")
    te = load_scene_split("test")
    tr_imgs, tr_tiles = tr["images"], boxes_to_tiles(tr["boxes"])
    te_imgs, te_tiles = te["images"], boxes_to_tiles(te["boxes"])
    print(f"Train: {tr_imgs.shape}  Test: {te_imgs.shape} "
          f"pos-rate train={tr_tiles.mean():.3f}")

    torch.manual_seed(42)
    model = ConvANFISSaliency()
    losses = train_stage12(model, tr_imgs, tr_tiles, epochs=40, batch=64,
                           lr=1e-3, device=dev)

    te_metrics = eval_stage12(model, te_imgs, te_tiles, device=dev)
    train_metrics = eval_stage12(model, tr_imgs, tr_tiles, device=dev)
    print("Test-Metriken:", te_metrics)
    print("Train-Metriken:", train_metrics)

    torch.save({"model": model.state_dict(),
                "metrics": te_metrics,
                "losses": losses},
               os.path.join(MODELS, "conv_anfis_saliency.pt"))

    # Loss-Verlauf
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, marker="o")
    ax.set(title="Stufe 1+2: Conv+ANFIS Saliency - Training-Loss",
           xlabel="Epoche", ylabel="BCE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "saliency_training_loss.png"), dpi=150)
    plt.close(fig)

    # Saliency-Karten auf Testbildern (3 Beispiele mit GT-Boxen)
    model.to(dev).eval()
    with torch.no_grad():
        X = torch.from_numpy(te_imgs[:6]).unsqueeze(1).to(dev)
        logits = model(X).cpu().numpy()          # [6,64]
        sal = 1.0 / (1.0 + numpy.exp(-logits))
    sal_map = sal.reshape(-1, 8, 8)              # [6,8,8]
    sal_img = numpy.kron(sal_map, numpy.ones((16, 16)))  # [6,128,128]

    fig, axes = plt.subplots(3, 2, figsize=(9, 13))
    for i in range(3):
        axes[i, 0].imshow(te_imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Szene {i}")
        axes[i, 1].imshow(te_imgs[i], cmap="gray", vmin=0, vmax=1)
        im = axes[i, 1].imshow(sal_img[i], cmap="jet", alpha=0.45,
                               vmin=sal.min(), vmax=sal.max())
        for k in range(te["labels"][i].shape[0]):
            x0, y0, x1, y1 = te["boxes"][i, k]
            if x0 < 0:
                continue
            axes[i, 1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                               fill=False, edgecolor="white", lw=1))
        axes[i, 1].set_title(f"Saliency + GT-Boxen (i={i})")
    fig.suptitle("Stufe 1+2: ANFIS-Saliency auf Test-Szenen")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "saliency_samples.png"), dpi=150)
    plt.close(fig)

    # CSV-Ergebnis
    import csv
    with open(os.path.join(RESULTS, "saliency_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "auc", "ap", "recall@prec0.5", "params"])
        w.writerow(["test", te_metrics["auc"], te_metrics["ap"],
                    te_metrics["recall@prec0.5"], 902])
        w.writerow(["train", train_metrics["auc"], train_metrics["ap"],
                    train_metrics["recall@prec0.5"], 902])
    print("Fertig. Ergebnisse in", RESULTS)


if __name__ == "__main__":
    main()