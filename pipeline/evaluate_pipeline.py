"""
Gesamt-Pipeline-Evaluation (Stufe 1-5) auf dem TEST-Scene-Split.

Pipeline je Bild:
  1+2. ConvStack+ANFIS -> Saliency 8x8   (fast, 902 Parameter)
  3.    Saliency>=0.5 -> Connected Components -> Regionen (gross->klein),
        grosse Regionen via hybrid2 Peak-seedete k-means-Subcluster
        (Decision Tree wurde per Ablation verworfen)
  4.    BNN (MC-Dropout) je Region: Klasse (0-9|10) + Box (cx,cy,w,h)
        * Full-Parse: alle Regionen bewerten
        * Early-Exit: nach der ersten sicheren Ziffern-Detektion stoppen
  5.    Bounding-Box = aus BNN-Box-Head in Bildkoordinaten umgerechnet

Baseline (Effizienz-/Qualitaetsvergleich):
  Alle 64 Kacheln einzeln (je Kachel-Crop 28x28) durch BNN bewerten,
  ohne Saliency-Gating.

Metriken:
  - Detektion: TP/FP (IoU>=0.5 gt-greedy), Recall, Precision,
    Bilder mit >=1 TP, Forward-Paesse (MC) pro Bild
  - Klassifikation: Acc auf TP-Detektionen; mit Confidence-Gating-Kurve
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

MC_S = 1
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
            sig_thr=0.05, gate=0.6):
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
        if p_max < gate:
            continue  # unsichere Kandidaten verwerfen (Confidenzgatter)
        dets.append({"box": box, "cls": cls, "conf": p_max, "sigma2": sig_max})
        if early_stop and cls != 10 and p_max >= conf_thr and sig_max <= sig_thr:
            stopped = True
            break
    return dets, fwd, stopped


def _refocus_patches(img, sal, regs, min_size=24, max_size=48):
    """Refokus-Crops: quadratisches Fenster um das Saliency-Zentrum der Region.

    Statt der rohen Tile-Bounding-Box (in der die Ziffer oft am Rand liegt)
    wird das Fenster um den saliency-gewichteten Schwerpunkt der Region
    zentriert -> Ziffer sitzt im Fensterzentrum. Groesse = Region-Ausdehnung,
    geclampt auf [min_size, max_size] (deckt die Trainings-Fenster 24..48px ab).
    """
    patches = []
    n_tile = 16
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        i0, j0 = x0 // n_tile, y0 // n_tile
        i1, j1 = (x1 - 1) // n_tile, (y1 - 1) // n_tile
        sub = sal[j0:j1 + 1, i0:i1 + 1]
        wsum = float(sub.sum())
        if wsum <= 0:
            continue
        ys, xs = numpy.indices(sub.shape)
        cy_t = float((sub * ys).sum() / wsum)
        cx_t = float((sub * xs).sum() / wsum)
        cx = x0 + (cx_t + 0.5) * n_tile
        cy = y0 + (cy_t + 0.5) * n_tile
        extent = max(x1 - x0, y1 - y0)
        w = int(min(max(extent, min_size), max_size))
        xc0 = int(numpy.clip(cx - w / 2, 0, 128 - w))
        yc0 = int(numpy.clip(cy - w / 2, 0, 128 - w))
        patches.append((img[yc0:yc0 + w, xc0:xc0 + w], xc0, yc0))
    return patches


def _kmeans_centroids(sal, k, mask=None, rng=None, max_iter=30,
                      init_centers=None):
    """Lloyd-k-means auf saliency-gewichteten Kachel-Zentren (8x8 Grid).

    Punkte = Zentren der aktiven Kacheln (gewichtete Position in Pixel),
    Gewicht = Saliency-Wert. Initialisierung via k-means++ ODER uebergebene
    Seeds (init_centers, Pixel-Koordinaten, k = len(init_centers)).
    Leere Cluster werden auf das vom verbleibenden Zentrum entfernteste
    Seed repariert (statt still verworfen zu werden).
    Liefert (centers [k,2] in Pixel, extents [k], assign [P], xs [P], ys [P]):
    assign/xs/ys sind die Lloyd-Zuordnung der aktiven Punkte.
    """
    if rng is None:
        rng = numpy.random.default_rng(42)
    if mask is None:
        mask = sal > 0
    ys, xs = numpy.nonzero(mask)
    if len(ys) == 0:
        return [], [], numpy.zeros(0, dtype=int), xs, ys
    w = sal[ys, xs]
    if w.sum() <= 0:
        return [], [], numpy.zeros(0, dtype=int), xs, ys
    pts = numpy.stack([xs * 16.0 + 8.0, ys * 16.0 + 8.0], axis=1).astype(numpy.float64)
    wts = w.astype(numpy.float64)
    if init_centers is not None:
        k = int(init_centers.shape[0])
        centers = numpy.asarray(init_centers, dtype=numpy.float64).copy()
        seeds = centers.copy()
    else:
        k = int(max(1, min(k, len(pts))))
        seeds = None
        # k-means++ Init
        centers = numpy.zeros((k, 2))
        centers[0] = pts[int(numpy.argmax(wts))]
        for c in range(1, k):
            mind = numpy.full(len(pts), numpy.inf)
            for prev in range(c):
                d2p = ((pts - centers[prev][None, :]) ** 2).sum(1)
                mind = numpy.minimum(mind, d2p)
            prob = wts * (mind + 1e-9)
            if prob.sum() <= 0:
                centers[c] = pts[c % len(pts)]
            else:
                centers[c] = pts[rng.choice(len(pts), p=prob / prob.sum())]
    # Lloyd-Iteration (gewichtete Zentren)
    assign = numpy.zeros(len(pts), dtype=int)
    for _ in range(max_iter):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        nassign = numpy.argmin(d, axis=1)
        for c in range(k):
            m = nassign == c
            if m.sum() == 0:
                if seeds is not None:
                    # weitestes verfuegbares Seed setzen
                    others = [j for j in range(k) if (nassign == j).sum() > 0]
                    if others:
                        ctrs = centers[others]
                        ds = ((seeds[:, None, :] - ctrs[None, :, :]) ** 2).sum(2)
                        centers[c] = seeds[int(numpy.argmax(ds.min(1)))]
                    else:
                        centers[c] = seeds[int(numpy.argmax(
                            ((seeds - centers[0][None, :]) ** 2).sum(1)))]
                else:
                    centers[c] = pts[numpy.argmax(((pts - centers[0][None, :]) ** 2).sum(1))]
            else:
                centers[c] = (pts[m] * wts[m][:, None]).sum(0) / wts[m].sum()
        if (nassign == assign).all():
            break
        assign = nassign
    extents = []
    for c in range(k):
        m = assign == c
        if m.sum() == 0:
            extents.append(0.0)
            continue
        extents.append(max((xs[m].max() - xs[m].min() + 1) * 16.0,
                           (ys[m].max() - ys[m].min() + 1) * 16.0))
    return centers, extents, assign, xs, ys


def _patches_from_squares(img, squares):
    """squares: (cx, cy, half, c) -> Crop-Patches (img, x0, y0).

    Fenster kantenexakt: x0 = floor(cx-half), x1 = ceil(cx+half), damit
    eigene Tiles vollstaendig drin sind und fremde Tiles nur beruehrt
    werden (Crop wird vom BNN ohnehin auf 28x28 resized).
    """
    patches = []
    for cx, cy, half, _ in squares:
        x0 = int(numpy.floor(cx - half))
        y0 = int(numpy.floor(cy - half))
        x1 = int(numpy.ceil(cx + half))
        y1 = int(numpy.ceil(cy + half))
        x0 = int(numpy.clip(x0, 0, 128))
        y0 = int(numpy.clip(y0, 0, 128))
        x1 = int(numpy.clip(x1, 0, 128))
        y1 = int(numpy.clip(y1, 0, 128))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches


def _dedup_squares(sal, squares, iou_thr=0.6, imp_thr=REGION_THR):
    """Redundante, ueberlappende Fenster entfernen (bei hoher Saliency-Masse).

    Sortiert nach im Fenster enthaltener Saliency-Masse absteigend, NMS mit
    IoU >= iou_thr: behalte das massenreichste Fenster, loesche ueberlappende.
    """
    if len(squares) <= 1:
        return squares
    scored = []
    for (cx, cy, half, c) in squares:
        x0 = int(numpy.clip(numpy.floor(cx - half), 0, 128))
        y0 = int(numpy.clip(numpy.floor(cy - half), 0, 128))
        x1 = int(numpy.clip(numpy.ceil(cx + half), 0, 128))
        y1 = int(numpy.clip(numpy.ceil(cy + half), 0, 128))
        wsum = float(sal[y0:y1, x0:x1].sum())
        scored.append((wsum, cx, cy, half, c))
    scored.sort(key=lambda s: -s[0])
    keep = []
    for s in scored:
        _, cx, cy, half, c = s
        x0 = int(numpy.clip(numpy.floor(cx - half), 0, 128))
        y0 = int(numpy.clip(numpy.floor(cy - half), 0, 128))
        x1 = int(numpy.clip(numpy.ceil(cx + half), 0, 128))
        y1 = int(numpy.clip(numpy.ceil(cy + half), 0, 128))
        dup = False
        for kx0, ky0, kx1, ky1 in keep:
            ix = max(0, min(x1, kx1) - max(x0, kx0))
            iy = max(0, min(y1, ky1) - max(y0, ky0))
            inter = ix * iy
            uni = (x1 - x0) * (y1 - y0) + (kx1 - kx0) * (ky1 - ky0) - inter
            if uni > 0 and inter / uni >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append((x0, y0, x1, y1))
            continue
    # nur behaltene Quadrate zurueckgeben
    out = []
    for s in scored:
        _, cx, cy, half, c = s
        x0 = int(numpy.clip(numpy.floor(cx - half), 0, 128))
        y0 = int(numpy.clip(numpy.floor(cy - half), 0, 128))
        x1 = int(numpy.clip(numpy.ceil(cx + half), 0, 128))
        y1 = int(numpy.clip(numpy.ceil(cy + half), 0, 128))
        if (x0, y0, x1, y1) in keep:
            out.append((cx, cy, half, c))
    return out


def _saliency_peaks(sal, mask=None, thr=REGION_THR, win=3):
    """Lokale Saliency-Maxima (Peaks) auf dem 8x8-Grid, nur Kacheln >= thr.

    Nur Kacheln in mask zaehlen (Nachbarschaftswerte ausserhalb mask werden
    ignoriert). Peaks in winxwin-Nachbarschaft absteigend nach Saliency,
    Gierig dedupliziert (max. ein Peak pro winxwin-Zelle).
    Rueckgabe: Liste von (y, x)-Tile-Indizes.
    """
    if mask is None:
        mask = numpy.ones_like(sal, dtype=bool)
    cand = numpy.argwhere(mask & (sal >= thr))
    if len(cand) == 0:
        return []
    vals = sal[cand[:, 0], cand[:, 1]]
    order = numpy.argsort(-vals)
    n = win // 2
    peaks = []
    for idx in order:
        yy, xx = int(cand[idx, 0]), int(cand[idx, 1])
        v = vals[idx]
        y0, y1 = max(0, yy - n), min(8, yy + n + 1)
        x0, x1 = max(0, xx - n), min(8, xx + n + 1)
        nb = sal[y0:y1, x0:x1].copy()
        nb[~mask[y0:y1, x0:x1]] = -numpy.inf
        nb[yy - y0, xx - x0] = -numpy.inf
        if v < numpy.max(nb):
            continue
        if all(abs(yy - py) > n or abs(xx - px) > n for py, px in peaks):
            peaks.append((yy, xx))
    return peaks


def _hybrid2_patches(img, sal, regs, big_extent=32, min_size=24, max_size=48,
                     orphan_size=28, peak_win=28, dedup_iou=0.75,
                     peak_dedup_iou=0.85, rng=None, imp_thr=REGION_THR):
    """hybrid2: Peak-seeded k-means mit garantierter Fenster-pro-Ziffer.

    Kleine Regionen -> Refokus-Crop (wie hybrid). Grosse Regionen:
    k = Anzahl lokaler Saliency-Peaks (statt Masse/1.5, kein Deckel),
    Peaks seeden die k-means-Zentren, leere Cluster werden repariert.
    Fenster sind auf die Cluster-Schwerpunkte zentriert (keine harte
    Fremd-Exklusion), Groesse = Cluster-Ausdehnung clampt auf
    [min_size, max_size].

    Kernpunkt (Fix gegen zu wenige Fenster): JEDER Peak erhaelt
    garantiert ein eigenes Peak-zentriertes Fenster (peak_win). Cluster-
    Fenster werden nur noch mit lockerem Schwellwert dedupliziert
    (dedup_iou), damit benachbarte Ziffern getrennte Fenster behalten;
    die finalen Peak-Fenster werden nur gegen nahezu identische
    Duplikate entfernt (peak_dedup_iou). -> ~1 Fenster/Ziffer statt
    ~1.5 Ziffern/Fenster.
    """
    patches = []
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        extent = max(x1 - x0, y1 - y0)
        if extent <= big_extent:
            patches.extend(_refocus_patches(img, sal, [r]))
            continue
        i0, j0 = x0 // 16, y0 // 16
        i1, j1 = (x1 - 1) // 16, (y1 - 1) // 16
        mask = numpy.zeros((8, 8), dtype=bool)
        mask[j0:j1 + 1, i0:i1 + 1] = sal[j0:j1 + 1, i0:i1 + 1] > 0
        peaks = _saliency_peaks(sal, mask=mask, thr=imp_thr)
        if len(peaks) == 0:
            patches.extend(_refocus_patches(img, sal, [r]))
            continue
        init = numpy.array([[px * 16.0 + 8.0, py * 16.0 + 8.0]
                            for py, px in peaks], dtype=numpy.float64)
        centers, extents, assign, xs, ys = _kmeans_centroids(
            sal, None, mask=mask, rng=rng, init_centers=init)
        squares = []
        for c, (cx, cy) in enumerate(centers):
            m = assign == c
            if m.sum() == 0:
                continue
            e = max((xs[m].max() - xs[m].min() + 1) * 16.0,
                    (ys[m].max() - ys[m].min() + 1) * 16.0)
            w = float(min(max(e, min_size), max_size))
            squares.append((cx, cy, w / 2.0, c))
        # lockere Konsolidierung: nur grob ueberlappende Fenster entfernen,
        # benachbarte Ziffern behalten ihre Fenster
        squares = _dedup_squares(sal, squares, iou_thr=dedup_iou,
                                 imp_thr=imp_thr)
        # garantiert ein Fenster je Peak (= je Ziffer)
        for py, px in peaks:
            squares.append((px * 16.0 + 8.0, py * 16.0 + 8.0,
                            peak_win / 2.0, -1.0))
        squares = _dedup_squares(sal, squares, iou_thr=peak_dedup_iou,
                                 imp_thr=imp_thr)
        patches.extend(_patches_from_squares(img, squares))
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


def _patches_for(img, sal, regs):
    """Crop-Strategie hybrid2: kleine CC-Regionen Refokus-Crop, grosse
    (verschmolzene) Regionen Peak-seedete k-means-Subcluster mit zentrierten
    Mindest-Fenstern und Waisen-Absicherung. Fruehere Strategien (bbox,
    refocus, kmeans, hybrid) wurden per Ablation verworfen (Archiv:
    Report/CropStrategien/crop_strategies_archive.py).
    """
    return _hybrid2_patches(img, sal, regs)


def evaluate_pipeline(bnn, sal_all, imgs, gtb, gtl, device="cuda",
                      early_stop=True, gate=0.6):
    tp = fp = cls_ok = cls_n = img_tp = 0
    fwd_s = 0.0
    for i in range(imgs.shape[0]):
        img, _, _, gt_idx = _semantic_scene(imgs, gtb, gtl, i)
        regs = _regions_from_saliency(sal_all[i])
        patches = _patches_for(img, sal_all[i], regs)
        dets, fwd, _ = _detect(bnn, patches, device, early_stop=early_stop,
                               gate=gate)
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


def _sample_indices(n, size, seed):
    rng = numpy.random.default_rng(seed)  # seed=None -> zufaellig
    return rng.choice(size, size=min(n, size), replace=False)


def _draw_detection_figure(img, gtb_i, gtl_i, gt_idx, dets, scene_id,
                           gate, title=""):
    """Zeichnet 2-Panel: GT-Boxen (gruen) vs. Pipeline-Detektionen (rot)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for a, t in [(axes[0], f"GT  (Szene {int(scene_id)})"),
                 (axes[1], f"Pipeline (gate={gate})")]:
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    for k in gt_idx:
        x0, y0, x1, y1 = gtb_i[k]
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, edgecolor="lime",
                                        lw=1.2, ls="--"))
        axes[0].text(x0, max(0, y0 - 1), str(int(gtl_i[k])),
                     color="lime", fontsize=8)
    for d in dets:
        if d["cls"] == 10:
            continue
        x0, y0, x1, y1 = d["box"]
        axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, edgecolor="red",
                                        lw=1.4))
        axes[1].text(x0, max(0, y0 - 1),
                     f"{d['cls']} {d['conf']:.2f}",
                     color="red", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def sample_detection_figures(saliency, bnn, imgs, gtb, gtl, device="cuda",
                             n=10, seed=None, gate=0.6):
    """Wie export_detection_samples, liefert aber die Figuren zurueck, statt
    PNGs zu speichern (fuer die Inline-Anzeige im Notebook)."""
    rng = numpy.random.default_rng(seed)
    pick = rng.choice(imgs.shape[0], size=min(n, imgs.shape[0]), replace=False)
    saliency.to(device).eval()
    bnn.to(device).eval()
    figs = []
    with torch.no_grad():
        for i in pick:
            img, gtb_i, gtl_i, gt_idx = _semantic_scene(imgs, gtb, gtl, int(i))
            X = torch.from_numpy(img[None, None].astype(numpy.float32)).to(device)
            sal = torch.sigmoid(saliency(X)).cpu().numpy().reshape(8, 8)
            regs = _regions_from_saliency(sal)
            patches = _patches_for(img, sal, regs)
            dets, fwd, _ = _detect(bnn, patches, device, early_stop=False,
                                   gate=gate)
            figs.append(_draw_detection_figure(img, gtb_i, gtl_i, gt_idx, dets,
                                               int(i), gate))
    return figs


def export_detection_samples(saliency, bnn, imgs, gtb, gtl, device="cuda",
                             n=10, seed=None, gate=0.6,
                             out_dir=None, tag="samples"):
    """Pipeline auf n zufaelligen Szenen laufen lassen und Boxen abspeichern.

    Nimmt n zufaellige Bilder aus dem Test-Split, laeuft die volle Kaskade
    (Saliency -> Regionen -> hybrid2-Fenster -> BNN+Box-Head), zeichnet pro
    Szene die GT-Boxen (gruen) und die Detektionen (rot, Klasse + Konfidenz)
    und speichert die PNGs nach out_dir/tag/. Ohne seed ist die Auswahl
    bei jedem Aufruf zufaellig.
    """
    if out_dir is None:
        out_dir = os.path.join(RESULTS, "samples")
    save_dir = os.path.join(out_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    figs = sample_detection_figures(saliency, bnn, imgs, gtb, gtl, device=device,
                                    n=n, seed=seed, gate=gate)
    files = []
    for fig, i in zip(figs, _sample_indices(n, imgs.shape[0], seed)):
        fn = os.path.join(save_dir, f"scene_{int(i)}.png")
        fig.savefig(fn, dpi=140)
        plt.close(fig)
        files.append(fn)
    print(f"[Export] {len(files)} zufaellige Szenen -> {save_dir}")
    return files


def confidence_curve(bnn, sal_all, imgs, gtb, gtl, device="cuda", n_max=400):
    """Acc/Koverage-Kurve bei Confidence-Gating (fuer MC-Dropout-Abstention)."""
    rows = []
    for i in range(min(n_max, imgs.shape[0])):
        img, gtb_i, gtl_i, gt_idx = _semantic_scene(imgs, gtb, gtl, i)
        regs = _regions_from_saliency(sal_all[i])
        patches = _patches_for(img, sal_all[i], regs)
        for patch, ox, oy in patches:
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

    # Nach der Messung: 10 zufaellige Bilder mit Boxen exportieren
    export_detection_samples(saliency, bnn, imgs, gtb, gtl, device=dev,
                             out_dir=RESULTS, tag="detections_mnist")

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