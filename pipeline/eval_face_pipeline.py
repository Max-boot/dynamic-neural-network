"""
Gesamt-Pipeline-Evaluation (Stufe 1-5) auf den WIDER-Validierungs-Szenen.

Pipeline je Bild: ConvANFIS-Saliency 8x8 -> Regionen -> hybrid2-Fenster ->
BNN(2 Klassen: 0=Gesicht, 1=BG) + Box-Head. Metriken wie MNIST-Pipeline:
TP/FP (IoU>=0.5), Precision, Recall, img_tp_rate, mean Forward-Paesse.
BG-Klasse = 1 (statt 10 bei Ziffern).
"""
import os
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, boxes_iou
from stage12 import ConvANFISSaliency
from bnn import BNN
from evaluate_pipeline import (_regions_from_saliency, _patches_for,
                               _crop_tensor)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SCENES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "wider_scenes")

MC_S = 1
REGION_THR = 0.5
BNN_C1, BNN_C2, BNN_HID = 40, 80, 192
BG_CLASS = 1


def _bnn_region(bnn, patch, device):
    xx = _crop_tensor(patch).to(device)
    mu, sig2, box_mu, _ = bnn.predict_mc_box(xx, S=MC_S, device=device)
    cls = int(mu[0].argmax())
    p_max = float(mu[0].max())
    cx, cy, w, h = box_mu[0].numpy()
    H0, W0 = patch.shape
    box = ((cx - w / 2) * W0, (cy - h / 2) * H0,
           (cx + w / 2) * W0, (cy + h / 2) * H0)
    return cls, p_max, box


def _detect(bnn, region_patches, device, gate=0.6):
    dets = []
    fwd = 0
    for patch, ox, oy in region_patches:
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        cls, p_max, box = _bnn_region(bnn, patch, device)
        fwd += MC_S
        box = (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)
        if cls == BG_CLASS and p_max >= 0.7:
            continue
        if p_max < gate:
            continue
        dets.append({"box": box, "conf": p_max})
    return dets, fwd


def _eval_detections(dets, img, gtb, gt_idx):
    used = set()
    tp = fp = 0
    for d in dets:
        best, bestk = 0.0, None
        for k in gt_idx:
            iou = boxes_iou(d["box"], gtb[k])
            if iou > best:
                best, bestk = iou, k
        if bestk is not None and best >= 0.5 and bestk not in used:
            used.add(bestk)
            tp += 1
        else:
            fp += 1
    return tp, fp


def evaluate_pipeline_face(bnn, sal_all, imgs, gtb, gtl, device="cuda",
                           gate=0.6):
    tp = fp = img_tp = 0
    fwd_s = 0.0
    for i in range(imgs.shape[0]):
        gt_idx = [k for k in range(gtb[i].shape[0]) if gtb[i, k, 0] >= 0]
        regs = _regions_from_saliency(sal_all[i])
        patches = _patches_for(imgs[i], sal_all[i], regs)
        dets, fwd = _detect(bnn, patches, device, gate=gate)
        fwd_s += fwd
        t, f = _eval_detections(dets, imgs[i], gtb[i], gt_idx)
        tp += t; fp += f
        if t > 0:
            img_tp += 1
    n_gt = int(sum(1 for g in gtl.reshape(-1) if g >= 0))
    n = imgs.shape[0]
    return {"n": n, "tp": tp, "fp": fp,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, n_gt),
            "img_tp": img_tp, "img_tp_rate": img_tp / n,
            "mean_fwd": fwd_s / n}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    saliency = ConvANFISSaliency()
    saliency.load_state_dict(torch.load(
        os.path.join(MODELS, "conv_anfis_saliency_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    bnn = BNN(n_class=2, box_head=True, c1=BNN_C1, c2=BNN_C2, hid=BNN_HID)
    bnn.load_state_dict(torch.load(
        os.path.join(MODELS, "bnn_mc_box_face.pt"),
        map_location="cpu", weights_only=False)["model"])

    te = load_scene_split("val", base=SCENES)
    imgs, gtb, gtl = te["images"], te["boxes"], te["labels"]
    print(f"Val-Szenen: {imgs.shape}")

    saliency.to(dev).eval()
    with torch.no_grad():
        X = torch.from_numpy(imgs).unsqueeze(1).to(dev)
        sal_all = []
        for i in range(0, X.shape[0], 64):
            sal_all.append(torch.sigmoid(saliency(X[i:i + 64]))
                           .cpu().numpy().reshape(-1, 8, 8))
        sal_all = numpy.concatenate(sal_all)

    for gate in (0.4, 0.5, 0.6, 0.7):
        m = evaluate_pipeline_face(bnn, sal_all, imgs, gtb, gtl, device=dev,
                                   gate=gate)
        print(f"--- gate={gate} ---")
        for k, v in m.items():
            print(f"  {k}: {round(v, 4)}" if isinstance(v, float) else f"  {k}: {v}")

    import csv
    with open(os.path.join(RESULTS, "face_pipeline_eval.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["gate", "metric", "value"])
        for gate in (0.4, 0.5, 0.6, 0.7):
            m = evaluate_pipeline_face(bnn, sal_all, imgs, gtb, gtl,
                                       device=dev, gate=gate)
            for k, v in m.items():
                w.writerow([gate, k, f"{float(v):.4f}"])
    print("Fertig ->", os.path.join(RESULTS, "face_pipeline_eval.csv"))


if __name__ == "__main__":
    main()