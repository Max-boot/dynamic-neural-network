"""
Vergleich: ConvMLPSaliency (Baseline MLP) vs ConvBottleneckSaliency (Linear Bottleneck).

Testet die MobileNetV2-Idee (Sandler et al.): Die projektive Schicht
(linear bottleneck) bleibt OHNE Aktivierung, um Manifold-Information in
niedrigdimensionalen Räumen zu bewahren.

Gleicher Trainings-Split, Seed und LR fuer fairen Vergleich.
Metriken: AUROC, AP, Recall@Precision>=0.5, Param-Zahl.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, boxes_to_tiles
from stage12 import (ConvMLPSaliency, ConvBottleneckSaliency,
                     train_stage12, eval_stage12)

SCENES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "wider_scenes")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

IN_CH = 3
EPOCHS = 40
SEED = 42


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # ---- Daten laden ----
    tr = load_scene_split("train", base=SCENES)
    te = load_scene_split("val", base=SCENES)
    tr_imgs, tr_tiles = tr["images"], boxes_to_tiles(tr["boxes"])
    te_imgs, te_tiles = te["images"], boxes_to_tiles(te["boxes"])
    print(f"Train: {tr_imgs.shape}  Val: {te_imgs.shape}  "
          f"pos-rate={tr_tiles.mean():.3f}")

    results = {}

    for name, ModelCls in [("mlp", ConvMLPSaliency),
                           ("bottleneck", ConvBottleneckSaliency)]:
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        model = ModelCls(in_ch=IN_CH)
        n_params = count_params(model)
        print(f"  Parameters: {n_params}  ({n_params*4} bytes f32)")

        t0 = time.time()
        losses = train_stage12(model, tr_imgs, tr_tiles, epochs=EPOCHS,
                               batch=64, lr=1e-3, device=dev)
        t_train = time.time() - t0

        met = eval_stage12(model, te_imgs, te_tiles, device=dev)
        print(f"  Stage12 Val: {met}")
        print(f"  Train time: {t_train:.1f}s")

        results[name] = {
            "losses": losses, "metrics": met,
            "n_params": n_params, "time": t_train,
        }

        # Checkpoint speichern
        torch.save({"model": model.state_dict(), "metrics": met,
                     "losses": losses, "n_params": n_params},
                    os.path.join(MODELS, f"saliency_{name}_exp.pt"))

    # ---- Vergleich ----
    print(f"\n{'='*60}")
    print("  VERGLEICH")
    print(f"{'='*60}")
    header = f"{'Metric':<22} {'MLP (Baseline)':>15} {'Bottleneck':>15} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for key in ["auc", "ap", "recall@prec0.5"]:
        v_mlp = results["mlp"]["metrics"][key]
        v_bot = results["bottleneck"]["metrics"][key]
        delta = v_bot - v_mlp
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<20} {v_mlp:>15.4f} {v_bot:>15.4f} {sign}{delta:>9.4f}")
    p_mlp = results["mlp"]["n_params"]
    p_bot = results["bottleneck"]["n_params"]
    print(f"  {'params':<20} {p_mlp:>15} {p_bot:>15} "
          f"{(p_bot-p_mlp):>+9}")

    # ---- Kurven ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, color in [("mlp", "C0"), ("bottleneck", "C1")]:
        ax[0].plot(results[name]["losses"], color=color, label=name)
    ax[0].set(title="Stage12 BCE Loss", xlabel="Epoche", ylabel="Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    keys = ["auc", "ap", "recall@prec0.5"]
    x = numpy.arange(len(keys))
    w = 0.35
    vals_mlp = [results["mlp"]["metrics"][k] for k in keys]
    vals_bot = [results["bottleneck"]["metrics"][k] for k in keys]
    ax[1].bar(x - w/2, vals_mlp, w, label="mlp", color="C0")
    ax[1].bar(x + w/2, vals_bot, w, label="bottleneck", color="C1")
    ax[1].set_xticks(x); ax[1].set_xticklabels(keys, rotation=15)
    ax[1].set(title="Validation Metriken", ylabel="Score")
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "saliency_bottleneck_exp.png"), dpi=150)
    plt.close(fig)

    print(f"\nErgebnisse -> {RESULTS}/saliency_bottleneck_exp.png")


if __name__ == "__main__":
    main()
