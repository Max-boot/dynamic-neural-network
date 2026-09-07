"""
Stufe 3: Decision Tree + Connected-Components -> Regionen-Selektion.

Tree-Features je Kachel (64 je Bild):
    [saliency (ANFIS-Sigmoid), mean, max, var]  (4 dims)
Label: Kachel ueberlappt mind. eine Ziffern-Box.

Der Tree wird RECALL-optimiert trainiert (positiv-Gewichtung, kontrollierte
Schwelle), damit keine relevanten Kacheln verloren gehen (Kaskaden-Schutz).
Danach: relevante Kachel-Maske -> Connected Components (4-Nachbarschaft)
         -> Regionen-Boxen, sortiert nach Groesse absteigend.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import (load_scene_split, boxes_to_tiles, extract_regions,
                         TILE)
from stage12 import ConvANFISSaliency

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def build_tree_features(model, imgs, device="cuda"):
    """Liefert [N*64, 4]: saliency, mean, max, var (instanz-normiert)."""
    import torch
    model.to(device).eval()
    X = torch.from_numpy(imgs).unsqueeze(1).to(device)
    sal, feats = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs]
            logits = model(xb)
            sal.append(torch.sigmoid(logits).cpu().numpy().reshape(-1, 1))
            feats.append(model.extract_features(xb).cpu().numpy().reshape(-1, 3))
    sal = numpy.concatenate(sal)
    feats = numpy.concatenate(feats)
    return numpy.concatenate([sal, feats], axis=1)


def region_recall(pred_regions_batch, gt_boxes_batch):
    """Anteil Ziffern, deren Box mit mind. einer predict-Region ueberlappt."""
    tot, hit = 0, 0
    for regions, boxes in zip(pred_regions_batch, gt_boxes_batch):
        for k in range(boxes.shape[0]):
            x0, y0, x1, y1 = boxes[k]
            if x0 < 0:
                continue
            tot += 1
            for r in regions:
                ix = min(r["x1"], x1) - max(r["x0"], x0)
                iy = min(r["y1"], y1) - max(r["y0"], y0)
                if ix > 0 and iy > 0:
                    hit += 1
                    break
    return hit / tot if tot else 0.0


def main():
    import pickle
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_recall_curve

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    model = ConvANFISSaliency()
    ckpt = __import__("torch").load(
        os.path.join(MODELS, "conv_anfis_saliency.pt"),
        map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    tr = load_scene_split("train")
    te = load_scene_split("test")
    Xtr = build_tree_features(model, tr["images"], device=dev)
    Xte = build_tree_features(model, te["images"], device=dev)
    ytr = boxes_to_tiles(tr["boxes"]).reshape(-1)
    yte = boxes_to_tiles(te["boxes"]).reshape(-1)
    print(f"Tree-Features: train {Xtr.shape}, test {Xte.shape}")

    tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=4,
                                  class_weight={0: 1.0, 1: 8.0},
                                  random_state=42)
    tree.fit(Xtr, ytr)
    with open(os.path.join(MODELS, "region_tree.pkl"), "wb") as f:
        pickle.dump(tree, f)

    # Schwellen-Kalibrierung auf TRAIN: max. Recall bei Precision >= 0.5
    p_tr = tree.predict_proba(Xtr)[:, 1]
    pr, rc, th = precision_recall_curve(ytr, p_tr)
    thr_full = numpy.concatenate([th, [1.0]])       # passend zu pr/rc
    cand = []
    for t, p_, r_ in zip(thr_full, pr, rc):
        if p_ >= 0.5:
            cand.append((r_, -t, t, p_))
    if cand:
        cand.sort(reverse=True)                      # hoechster Recall zunaechst
        best_rec, _, t_hold, _ = cand[0]
        t_hold = float(t_hold)
        print(f"Threshold (train-validiert, Recall-Ziel): {t_hold:.3f}  "
              f"Train-Recall@{t_hold:.3f}={best_rec:.3f}")
    else:
        t_hold = 0.3
        best_rec = 0.0
        print(f"WARN: kein Threshold mit Precision>=0.5; Fallback {t_hold}")

    # Tree-Metriken auf TEST bei t_hold
    p_te = tree.predict_proba(Xte)[:, 1]
    pr_te, rc_te, th_te = precision_recall_curve(yte, p_te)
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_te = roc_auc_score(yte, p_te)
    ap_te = average_precision_score(yte, p_te)
    thr_full_te = numpy.concatenate([th_te, [1.0]])
    m0 = numpy.where(thr_full_te >= t_hold)[0][0]
    rec_at_hold = float(rc_te[m0])
    pr_at_hold = float(pr_te[m0 - 1] if m0 > 0 else pr_te[0])

    # Regionen je Testbild bei t_hold
    pred_masks = (p_te.reshape(-1, 8, 8) >= t_hold)
    regions_per_img = []
    all_regions = []
    for i in range(pred_masks.shape[0]):
        regions, _ = extract_regions(pred_masks[i])
        all_regions.append(regions)
        regions_per_img.append(len(regions))
    rec_region = region_recall(all_regions, te["boxes"])
    print(f"Test Tree: AUC={auc_te:.4f} AP={ap_te:.4f} "
          f"Recall@{t_hold:.2f}={rec_at_hold:.3f} Prec={pr_at_hold:.3f} "
          f"Region-Recall={rec_region:.3f} "
          f"Regionen/Bild={numpy.mean(regions_per_img):.1f}")

    with open(os.path.join(RESULTS, "stage3_results.csv"), "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["threshold", f"{t_hold:.4f}"])
        w.writerow(["auc", f"{auc_te:.4f}"])
        w.writerow(["ap", f"{ap_te:.4f}"])
        w.writerow(["recall@thr", f"{rec_at_hold:.4f}"])
        w.writerow(["precision@thr", f"{pr_at_hold:.4f}"])
        w.writerow(["region_recall", f"{rec_region:.4f}"])
        w.writerow(["regions_per_image_mean", f"{numpy.mean(regions_per_img):.2f}"])

    # Plot: PR-Kurve + Regionen auf 3 Testbildern
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rc_te, pr_te, lw=2)
    ax[0].set(title="Tree PR-Kurve (Test)", xlabel="Recall", ylabel="Precision")
    ax[0].axhline(0.5, color="gray", ls="--", lw=1)
    ax[0].grid(alpha=0.3)
    for i in range(2):
        img = te["images"][i]
        regs = all_regions[i]
        ax[1].imshow(img, cmap="gray", vmin=0, vmax=1)
        for r in regs:
            ax[1].add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (r["x0"], r["y0"]), r["x1"] - r["x0"], r["y1"] - r["y0"],
                fill=False, edgecolor="lime", lw=1.2))
        for k in range(te["labels"][i].shape[0]):
            x0, y0, x1, y1 = te["boxes"][i, k]
            if x0 < 0:
                continue
            ax[1].add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=0.8))
        ax[1].set_title(f"Regionen (gruen) vs GT (rot), Bild {i}")
    fig.suptitle("Stufe 3: Decision Tree Regionen")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "stage3_regions.png"), dpi=150)
    plt.close(fig)
    print("Fertig. Ergebnis unter", RESULTS)


if __name__ == "__main__":
    main()