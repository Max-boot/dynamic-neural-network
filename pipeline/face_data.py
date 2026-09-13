"""
Datensatz fuer das Gesichts-BNN (Stufe 4+5): 1 Objektklasse (Gesicht) + Box.

Crop-Verteilung identisch zur MNIST-Pipeline: Fenster/Tile/negative Samples
aus 128x128-Szenen, bilinear auf 28x28 resized. Klassen:
  - 0 = Gesicht
  - 1 = Hintergrund  (BG_CLASS)
Box-Ziele (cx,cy,w,h) nur fuer Klasse 0; sonst Dummy (0.5,0.5,0,0).

Keine Cluttered-Augmentation (war Ziffern-spezifisch). Aus Szenen geladen
via base="wider_scenes" (identisches Format wie scene_dataset).
"""
import os

import numpy
import torch
from torch.utils.data import TensorDataset

from data_common import load_scene_split, crop

BG_CLASS = 1
SCENE = 128


def near_any(box, boxes, margin=6):
    x0, y0, x1, y1 = box
    for b in boxes:
        if b[0] < 0:
            continue
        if not (x1 < b[0] - margin or x0 > b[2] + margin or
                y1 < b[1] - margin or y0 > b[3] + margin):
            return True
    return False


def window_samples(imgs, boxes, labels, rng, max_per_image,
                   win_range=(24, 48), offset=8, max_foreign=4):
    """Fenster-Crops um Gesichter (wie bnn_data._window_samples)."""
    xs, ys, bs = [], [], []
    for n in range(imgs.shape[0]):
        ks = numpy.where(boxes[n, :, 0] >= 0)[0]
        for _ in range(max_per_image):
            if len(ks) == 0:
                break
            k = int(rng.choice(ks))
            gx0, gy0, gx1, gy1 = boxes[n, k]
            lbl = int(labels[n, k])
            ws = int(rng.integers(win_range[0], win_range[1] + 1))
            cx = int((gx0 + gx1) / 2) + int(rng.integers(-offset, offset + 1))
            cy = int((gy0 + gy1) / 2) + int(rng.integers(-offset, offset + 1))
            wx0 = max(0, int(cx - ws / 2))
            wy0 = max(0, int(cy - ws / 2))
            wx0 = min(wx0, SCENE - ws)
            wy0 = min(wy0, SCENE - ws)
            wx1 = wx0 + ws
            wy1 = wy0 + ws
            foreign = 0
            for m in ks:
                if m == k:
                    continue
                b = boxes[n, m]
                ix = max(0, min(b[2], wx1) - max(b[0], wx0))
                iy = max(0, min(b[3], wy1) - max(b[1], wy0))
                if ix > 0 and iy > 0:
                    foreign += 1
            if foreign > max_foreign:
                continue
            patch = imgs[n, wy0:wy1, wx0:wx1]
            if patch.shape[0] < 6 or patch.shape[1] < 6:
                continue
            xs.append(crop(patch, 28))
            ys.append(lbl)
            bs.append((max(gx0, wx0), max(gy0, wy0),
                       min(gx1, wx1), min(gy1, wy1), wx0, wy0, ws, ws))
    return xs, ys, bs


def tile_samples(imgs, boxes, labels, rng, max_per_image):
    """Dominante Kachel-Crops (16x16) wie bnn_data._tile_samples."""
    xs, ys, bs = [], [], []
    for n in range(imgs.shape[0]):
        ks = numpy.where(boxes[n, :, 0] >= 0)[0]
        for _ in range(max_per_image):
            if len(ks) == 0:
                break
            k = int(rng.choice(ks))
            gx0, gy0, gx1, gy1 = boxes[n, k]
            lbl = int(labels[n, k])
            cx = int((gx0 + gx1) / 2)
            cy = int((gy0 + gy1) / 2)
            tx0 = (cx // 16) * 16
            ty0 = (cy // 16) * 16
            tx1 = min(SCENE, tx0 + 16)
            ty1 = min(SCENE, ty0 + 16)
            patch = imgs[n, ty0:ty1, tx0:tx1]
            xs.append(crop(patch, 28))
            ys.append(lbl)
            bs.append((max(gx0, tx0), max(gy0, ty0),
                       min(gx1, tx1), min(gy1, ty1), tx0, ty0, 16, 16))
    return xs, ys, bs


def negative_samples(imgs, boxes, rng, n_per_image, size_range=(10, 30),
                     margin=8):
    xs, ys, bs = [], [], []
    for n in range(imgs.shape[0]):
        attempts = 0
        placed = 0
        while placed < n_per_image and attempts < n_per_image * 20:
            attempts += 1
            s = int(rng.integers(size_range[0], size_range[1] + 1))
            x0 = int(rng.integers(0, max(1, SCENE - s)))
            y0 = int(rng.integers(0, max(1, SCENE - s)))
            if near_any((x0, y0, x0 + s, y0 + s), boxes[n], margin=margin):
                continue
            patch = imgs[n, y0:y0 + s, x0:x0 + s]
            xs.append(crop(patch, 28))
            ys.append(BG_CLASS)
            bs.append((0, 0, 0, 0, x0, y0, s, s))
            placed += 1
    return xs, ys, bs


def normalize_boxes(items, n):
    out = numpy.zeros((n, 4), dtype=numpy.float32)
    for i, (gx0, gy0, gx1, gy1, wx0, wy0, ww, wh) in enumerate(items):
        cx = (gx0 + gx1) / 2 - wx0
        cy = (gy0 + gy1) / 2 - wy0
        bw = gx1 - gx0
        bh = gy1 - gy0
        out[i] = (cx / ww, cy / wh, bw / ww, bh / wh)
    return out


def build_face_datasets(seed=42, base=None, max_pos_window=3, max_pos_tile=3,
                        neg_per_img=6):
    """Train/Test-Tensordatasets (x[3,28,28], y_cls[2], y_box[4])."""
    if base is None:
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "wider_scenes")
    rng = numpy.random.default_rng(seed)
    tr = load_scene_split("train", base=base)
    te = load_scene_split("val", base=base)

    tr_wx, tr_ws, tr_wb = window_samples(tr["images"], tr["boxes"],
                                         tr["labels"], rng, max_pos_window)
    tr_tx, tr_ts, tr_tb = tile_samples(tr["images"], tr["boxes"],
                                       tr["labels"], rng, max_pos_tile)
    tr_nx, tr_ns, tr_nb = negative_samples(tr["images"], tr["boxes"], rng,
                                           neg_per_img)

    te_wx, te_ws, te_wb = window_samples(te["images"], te["boxes"],
                                         te["labels"], rng, max_pos_window)
    te_tx, te_ts, te_tb = tile_samples(te["images"], te["boxes"],
                                       te["labels"], rng, max_pos_tile)
    te_nx, te_ns, te_nb = negative_samples(te["images"], te["boxes"], rng,
                                           neg_per_img)

    def pack(xs, ys, bs):
        X = numpy.stack(xs).astype(numpy.float32)
        if X.ndim == 4:                    # [N,28,28,C] channel-last -> [N,C,28,28]
            X = numpy.transpose(X, (0, 3, 1, 2))
        Y = numpy.asarray(ys, dtype=numpy.int64)
        B = normalize_boxes(bs, len(Y))
        B[Y == BG_CLASS] = (0.5, 0.5, 0.0, 0.0)
        return X, Y, B

    X_tr, y_tr, b_tr = pack(tr_wx + tr_tx + tr_nx, tr_ws + tr_ts + tr_ns,
                            tr_wb + tr_tb + tr_nb)
    X_te, y_te, b_te = pack(te_wx + te_tx + te_nx, te_ws + te_ts + te_ns,
                            te_wb + te_tb + te_nb)
    return (TensorDataset(torch.from_numpy(X_tr),
                          torch.from_numpy(y_tr), torch.from_numpy(b_tr)),
            TensorDataset(torch.from_numpy(X_te),
                          torch.from_numpy(y_te), torch.from_numpy(b_te)))


if __name__ == "__main__":
    import collections
    tr, te = build_face_datasets(max_pos_window=2, max_pos_tile=2,
                                 neg_per_img=3)
    ytr = [int(tr[i][1]) for i in range(0, len(tr), 971)]
    yte = [int(te[i][1]) for i in range(0, len(te), 751)]
    print("Train:", len(tr), "Val:", len(te))
    print("Train-Klassen:", collections.Counter(ytr))
    print("Val-Klassen :", collections.Counter(yte))
    print("Box-Beispiel:", tr[0][2].tolist(), "BG:",  tr[-1][2].tolist())