"""
Gesamt-Pipeline-Evaluation (Stufe 1-5) auf dem TEST-Scene-Split.

Pipeline je Bild:
  1+2. ConvStack+ANFIS -> Saliency 8x8   (fast, 902 Parameter)
  3.    Saliency>=0.5 -> Connected Components -> Regionen (gross->klein)
        (Decision Tree wird separat als Ablation berichtet)
  4.    BNN (MC-Dropout) je Region: Klasse (0-9|10) + Box (cx,cy,w,h)
        * Full-Parse: alle Regionen bewerten
        * Early-Exit: nach der ersten sicheren Ziffern-Detektion stoppen
  5.    Bounding-Box = aus BNN-Box-Head in Bildkoordinaten umgerechnet

Baseline (Effizienz-/Qualitaetsvergleich):
  Alle 64 Kacheln einzeln (je Kachel-Crop 28x28) durch BNN bewerten,
  ohne Saliency/Tree-Gating.

Metriken:
  - Detektion: TP/FP (IoU>=0.5 gt-greedy), Recall, Precision,
    Bilder mit >=1 TP, Forward-Paesse (MC) pro Bild
  - Klassifikation: Acc auf TP-Detektionen; mit Confidence-Gating-Kurve
  - Tree-Ablation: Region-Recall, Regionen/Bild
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, extract_regions, crop, boxes_iou
from stage12 import ConvANFISSaliency, eval_stage12
from bnn import BNN

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MC_S = 8
REGION_THR = 0.5
BNN_C1, BNN_C2, BNN_HID = 40, 80, 192


def _crop_tensor(patch):
    return torch.from_numpy(crop(patch, 28).copy()).unsqueeze(0).unsqueeze(0)


def _regions_from_saliency(sal):
    regs, _ = extract_regions(sal >= REGION_THR)
    regs.sort(key=lambda r: -r["tiles"])
    return regs


def _bnn_region(bnn, patch, device):
    """patch (H,W,float) -> (cls, p_max, sig2_max, box_abs)."""
    xx = _crop_tensor(patch).to(device)
    mu, sig2, box_mu, box_var = bnn.predict_mc_box(xx, S=MC_S, device=device)
    cls = int(mu[0].argmax())
    p_max = float(mu[0].max())
    sig_max = float(sig2[0].max())
    cx, cy, w, h = box_mu[0].numpy()
    H0, W0 = patch.shape
    box = (cx - w / 2) * W0, (cy - h / 2) * H0, (cx + w / 2) * W0, (cy + h / 2) * H0
    return cls, p_max, sig_max, box


def _semantic_scene(imgs, gt_boxes, gt_labels, i):
    img = imgs[i]
    gtb = gt_boxes[i]
    gtl = gt_labels[i]
    gt_idx = [k for k in range(gtb.shape[0]) if gtb[k, 0] >= 0]
    return img, gtb, gtl, gt_idx


def _detect(bnn, region_patches, device, early_stop=True, conf_thr=0.9,
            sig_thr=0.05):
    """Regionen -> Liste von Detektionen + Anzahl Forward-Paesse."""
    dets = []
    fwd = 0
    stopped = False
    for patch, ox, oy in region_patches:
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        cls, p_max, sig_max, box = _bnn_region(bnn, patch, device)
        fwd += MC_S
        box = (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)
        if cls == 10 and p_max >= 0.7:
            continue  # sicher Hintergrund -> verwerfen (kein FP-Eintrag)
        if p_max < 0.6:
            continue  # unsichere Kandidaten verwerfen (Confidenzgatter)
        dets.append({"box": box, "cls": cls, "conf": p_max, "sigma2": sig_max})
        if early_stop and cls != 10 and p_max >= conf_thr and sig_max <= sig_thr:
            stopped = True
            break
    return dets, fwd, stopped


def _region_patches(img, regs, margin=2):
    patches = []
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
        x1 = min(128, x1 + margin); y1 = min(128, y1 + margin)
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches


def _eval_detections(dets, img, gtb, gtl, gt_idx):
    used = set()
    tp = fp = 0
    cls_ok = 0
    cls_n = 0
    for d in dets:
        best, bestk = 0.0, None
        for k in gt_idx:
            iou = boxes_iou(d["box"], gtb[k])
            if iou > best:
                best, bestk = iou, k
        if bestk is not None and best >= 0.5 and bestk not in used:
            used.add(bestk)
            tp += 1
            cls_n += 1
            if d["cls"] == int(gtl[bestk]):
                cls_ok += 1
        else:
            fp += 1
    return tp, fp, cls_ok, cls_n


def evaluate_pipeline(bnn, sal_all, imgs, gtb, gtl, device="cuda",
                      early_stop=True):
    tp = fp = cls_ok = cls_n = img_tp = 0
    fwd_s = 0.0
    for i in range(imgs.shape[0]):
        img, _, _, gt_idx = _semantic_scene(imgs, gtb, gtl, i)
        regs = _regions_from_saliency(sal_all[i])
        patches = _region_patches(img, regs)
        dets, fwd, _ = _detect(bnn, patches, device, early_stop=early_stop)
        fwd_s += fwd
        t, f, cok, cn = _eval_detections(dets, img, gtb[i], gtl[i], gt_idx)
        tp += t; fp += f; cls_ok += cok; cls_n += cn
        if t > 0:
            img_tp += 1
    n = imgs.shape[0]
    return {"n": n,
            "tp": tp, "fp": fp,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, _count_gt(gtl)),
            "img_tp": img_tp, "img_tp_rate": img_tp / n,
            "cls_acc": cls_ok / max(1, cls_n),
            "mean_fwd": fwd_s / n}


def _count_gt(gtl):
    return int(sum(1 for g in gtl.reshape(-1) if g >= 0))


def baseline_tiles(bnn, imgs, gtb, gtl, device="cuda"):
    """Baseline: jede einzelne Kachel (16x16 -> 28x28) durch BNN."""
    tp = fp = cls_ok = cls_n = img_tp = 0
    n = imgs.shape[0]
    for i in range(n):
        img, _, _, gt_idx = _semantic_scene(imgs, gtb, gtl, i)
        dets = []
        for j0 in range(0, 128, 16):
            for i0 in range(0, 128, 16):
                patch = img[j0:j0 + 16, i0:i0 + 16]
                cls, p_max, sig_max, box = _bnn_region(bnn, patch, device)
                if cls == 10 and p_max >= 0.7:
                    continue
                box = (box[0] + i0, box[1] + j0, box[2] + i0, box[3] + j0)
                dets.append({"box": box, "cls": cls, "conf": p_max,
                             "sigma2": sig_max})
        t, f, cok, cn = _eval_detections(dets, img, gtb[i], gtl[i], gt_idx)
        tp += t; fp += f; cls_ok += cok; cls_n += cn
        if t > 0:
            img_tp += 1
    return {"n": n, "tp": tp, "fp": fp,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, _count_gt(gtl)),
            "img_tp": img_tp, "img_tp_rate": img_tp / n,
            "cls_acc": cls_ok / max(1, cls_n),
            "mean_fwd": 64 * MC_S}


def confidence_curve(bnn, sal_all, imgs, gtb, gtl, device="cuda", n_max=400):
    """Acc/Koverage-Kurve bei Confidence-Gating (fuer MC-Dropout-Abstention)."""
    rows = []
    for i in range(min(n_max, imgs.shape[0])):
        img, gtb_i, gtl_i, gt_idx = _semantic_scene(imgs, gtb, gtl, i)
        regs = _regions_from_saliency(sal_all[i])
        for patch, ox, oy in _region_patches(img, regs):
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue
            cls, p_max, sig_max, box = _bnn_region(bnn, patch, device)
            box = (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)
            best, bestk = 0.0, None
            for k in gt_idx:
                iou = boxes_iou(box, gtb_i[k])
                if iou > best:
                    best, bestk = iou, k
            rows.append({"p": p_max, "sig": sig_max, "tp": bestk is not None
                         and best >= 0.5, "ok": bestk is not None
                         and best >= 0.5 and cls == int(gtl_i[bestk]), "bg": cls == 10})
    out = []
    for t in numpy.arange(0.2, 0.99, 0.05):
        sel = [r for r in rows if r["p"] >= t]
        if len(sel) == 0:
            continue
        acc = sum(r["ok"] for r in sel) / len(sel)
        det = sum(r["tp"] for r in sel) / len(sel)
        out.append({"t": float(t), "acc": acc, "det": det, "cov": len(sel)})
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    os.makedirs(RESULTS, exist_ok=True)

    saliency = ConvANFISSaliency()
    saliency.load_state_dict(torch.load(
        os.path.join(MODELS, "conv_anfis_saliency.pt"),
        map_location="cpu", weights_only=False)["model"])
    bnn = BNN(n_class=11, box_head=True, c1=BNN_C1, c2=BNN_C2, hid=BNN_HID)
    bnn.load_state_dict(torch.load(
        os.path.join(MODELS, "bnn_mc_box.pt"),
        map_location="cpu", weights_only=False)["model"])

    te = load_scene_split("test")
    imgs, gtb, gtl = te["images"], te["boxes"], te["labels"]

    # 1) Stage1+2 Tile-Metriken
    tiles = _boxes_to_tiles(gtb[:500])
    s_m = eval_stage12(saliency.to(dev), imgs[:500], tiles, device=dev)
    print("Stage1+2:", {k: round(v, 3) for k, v in s_m.items()})

    # Saliency fuer alle Bilder vorberechnen
    saliency.to(dev).eval()
    with torch.no_grad():
        X = torch.from_numpy(imgs).unsqueeze(1).to(dev)
        sal_all = []
        for i in range(0, X.shape[0], 64):
            sal_all.append(torch.sigmoid(saliency(X[i:i + 64]))
                           .cpu().numpy().reshape(-1, 8, 8))
        sal_all = numpy.concatenate(sal_all)

    # 2) Pipeline full-parse + early-exit
    print("Pipeline full-parse ...")
    m_full = evaluate_pipeline(bnn.to(dev), sal_all, imgs, gtb, gtl,
                               device=dev, early_stop=False)
    print("Pipeline early-exit ...")
    m_early = evaluate_pipeline(bnn, sal_all, imgs, gtb, gtl,
                                device=dev, early_stop=True)

    # 3) Baseline
    print("Baseline (alle 64 Kacheln) ...")
    m_base = baseline_tiles(bnn, imgs[:400], gtb[:400], gtl[:400], device=dev)

    # 4) Confidence-Gating-Kurve
    print("Confidence-Kurve ...")
    curve = confidence_curve(bnn, sal_all, imgs, gtb, gtl, device=dev)

    for name, m in [("FULL", m_full), ("EARLY", m_early), ("BASELINE", m_base)]:
        print(f"--- {name} ---")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    import csv
    with open(os.path.join(RESULTS, "pipeline_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "value"])
        for name, m in [("stage12", dict(s_m)),
                        ("pipeline_full", m_full),
                        ("pipeline_early", m_early),
                        ("baseline_tiles", m_base)]:
            for k, v in m.items():
                w.writerow([name, k, f"{float(v):.4f}" if isinstance(v, (int, float)) else v])
        w.writerow([])
        w.writerow(["confidence_curve", "threshold", "cls_acc", "det_prec", "samples"])
        for c in curve:
            w.writerow(["confidence_curve", f"{c['t']:.2f}", f"{c['acc']:.4f}",
                        f"{c['det']:.4f}", c["cov"]])

    # Plot 1: Effizienz vs Detektion
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["Pipeline\nfull", "Pipeline\nearly", "Baseline\n(64 Kacheln)"]
    fwds = [m_full["mean_fwd"], m_early["mean_fwd"], m_base["mean_fwd"]]
    rates = [m_full["img_tp_rate"], m_early["img_tp_rate"], m_base["img_tp_rate"]]
    x = numpy.arange(3)
    ax[0].bar(x - 0.2, fwds, width=0.4, label="Forward-Pässe/Bild (MC×8)")
    ax[0].bar(x + 0.2, rates, width=0.4, label="Bilder mit ≥1 Detektion")
    ax[0].set_xticks(x, names)
    ax[0].set_title("Effizienz vs. Detektions-Rate")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].plot([c["t"] for c in curve], [c["acc"] for c in curve],
               marker="o", label="Klasse (0-9) Acc bei conf≥t")
    ax[1].plot([c["t"] for c in curve], [c["det"] for c in curve],
               marker="s", label="IoU≥0.5 (Kandidat korrekt)")
    ax[1].set_xlabel("Confidence-Schwelle t (max softmax)")
    ax[1].set_ylabel("Verhaeltnis")
    ax[1].set_title("MC-Dropout: Confidenz-Gating (Abstention)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pipeline_vs_baseline.png"), dpi=150)
    print("Fertig ->", RESULTS)


def _boxes_to_tiles(boxes):
    from data_common import GRID, TILE as T
    N, K, _ = boxes.shape
    out = numpy.zeros((N, GRID, GRID), dtype=numpy.int64)
    for n in range(N):
        for k in range(K):
            x0, y0, x1, y1 = boxes[n, k]
            if x0 < 0:
                continue
            a = int(max(0, y0 // T)); b = int(min(GRID, (y1 - 1) // T + 1))
            c = int(max(0, x0 // T)); d = int(min(GRID, (x1 - 1) // T + 1))
            out[n, a:b, c:d] = 1
    return out


if __name__ == "__main__":
    main()