"""
Datensatz fuer das Stufe-4/5-BNN (MC-Dropout + Bounding-Box-Head).

Der BNN-Crop muss derselben Verteilung folgen wie die Region-Crops in der
Pipeline (Kachel-Region der Stufe 3 -> bilinear auf 28x28).

Jedes Sample: (x [1,28,28], y_cls [0-9|10], y_box [4]) mit normalisierten
Box-Zielen (cx, cy, w, h) relativ zum Crop-Fenster im Originalbild.
Für Hintergrund-Samples (Klasse 10) ist y_box = [0.5, 0.5, 0, 0] (wird im
Training maskiert/ignoriert).

Positiv:
  - Fenster-Crops um GT-Boxen (mit Jitter/Versatz 0..8px), Fenstergroesse
    24..48px -> deckt Peak-Offsets der Inferenz ab; Box-Ziel = GT in Fenster
  - dominante Kachel (16x16) je Ziffer -> entspricht Region-Crop
  - Cluttered-Center-Crops (einige) als Stil-Augmentation (Box = ganzes Bild)

Negativ:
  - random Szenen-Patches, die zu keiner GT-Box nah sind (Randabstand)
"""
import os

import numpy
import torch
from torch.utils.data import TensorDataset

from data_common import PROJECT_ROOT, load_scene_split, crop

BG_CLASS = 10


def _near_any(box, boxes, margin=6):
    x0, y0, x1, y1 = box
    for b in boxes:
        if b[0] < 0:
            continue
        if not (x1 < b[0] - margin or x0 > b[2] + margin or
                y1 < b[1] - margin or y0 > b[3] + margin):
            return True
    return False


def _window_samples(imgs, boxes, labels, rng, max_per_image,
                    win_range=(24, 48), offset=8, max_foreign=3):
    """Fenster-Crops um Ziffern; gibt (crop28, cls, box_target_abs) zurueck.

    max_foreign begrenzt die Ueberfuellung: Benachbarte GT-Ziffern, die das
    Fenster (anteilig) ueberlappen, werden mitgezaehlt. Ein Sample wird
    verworfen, wenn mehr als max_foreign fremde Ziffern im Fenster liegen
    (Ziel: 1-4 Ziffern pro Crop statt bis zu ~9 bei dicht gepackten Szenen).
    """
    xs, ys, bs = [], [], []
    n_img = imgs.shape[0]
    for n in range(n_img):
        ks = numpy.where(boxes[n, :, 0] >= 0)[0]
        for _ in range(max_per_image):
            if len(ks) == 0:
                break
            k = int(rng.choice(ks))
            gx0, gy0, gx1, gy1 = boxes[n, k]
            lbl = int(labels[n, k])
            # Fenstergroesse zufaellig
            ws = int(rng.integers(win_range[0], win_range[1] + 1))
            # Fensterzentrum zufaellig um GT-Zentrum versetzt
            cx = int((gx0 + gx1) / 2) + int(rng.integers(-offset, offset + 1))
            cy = int((gy0 + gy1) / 2) + int(rng.integers(-offset, offset + 1))
            wx0 = max(0, int(cx - ws / 2))
            wy0 = max(0, int(cy - ws / 2))
            wx0 = min(wx0, 128 - ws)
            wy0 = min(wy0, 128 - ws)
            wx1 = wx0 + ws
            wy1 = wy0 + ws
            # Ueberfuellung pruefen: fremde GT-Ziffern im Fenster
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
            # Box-Ziel in Originalkoordinaten (fuer spaetere Normalisierung)
            bs.append((max(gx0, wx0), max(gy0, wy0),
                       min(gx1, wx1), min(gy1, wy1), wx0, wy0, ws, ws))
    return xs, ys, bs


def _tile_samples(imgs, boxes, labels, rng, max_per_image):
    """Dominante Kachel-Crops (16x16) -> Region-Crop-Verteilung."""
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
            tx1 = min(128, tx0 + 16)
            ty1 = min(128, ty0 + 16)
            patch = imgs[n, ty0:ty1, tx0:tx1]
            xs.append(crop(patch, 28))
            ys.append(lbl)
            bs.append((max(gx0, tx0), max(gy0, ty0),
                       min(gx1, tx1), min(gy1, ty1), tx0, ty0, 16, 16))
    return xs, ys, bs


def _negative_samples(imgs, boxes, rng, n_per_image, size_range=(10, 30),
                      margin=8):
    xs, ys, bs = [], [], []
    for n in range(imgs.shape[0]):
        attempts = 0
        placed = 0
        while placed < n_per_image and attempts < n_per_image * 20:
            attempts += 1
            s = int(rng.integers(size_range[0], size_range[1] + 1))
            x0 = int(rng.integers(0, max(1, 128 - s)))
            y0 = int(rng.integers(0, max(1, 128 - s)))
            if _near_any((x0, y0, x0 + s, y0 + s), boxes[n], margin=margin):
                continue
            patch = imgs[n, y0:y0 + s, x0:x0 + s]
            xs.append(crop(patch, 28))
            ys.append(BG_CLASS)
            bs.append((0, 0, 0, 0, x0, y0, s, s))  # Box ignoriert (maskiert)
            placed += 1
    return xs, ys, bs


def _clutter_crops(n_total, rng):
    """Cluttered-Center-Crops [N,1,28,28] als Augmentation (Box = Bild)."""
    base = os.path.join(PROJECT_ROOT, "cluttered_mnist")
    xs, ys, bs = [], [], []
    for split, n_used in [("train", n_total)]:
        path = os.path.join(base, f"cluttered_{split}.npz")
        if not os.path.exists(path):
            continue
        d = numpy.load(path)
        imgs = numpy.asarray(d["images"])
        labels = numpy.asarray(d["digit_labels"])  # [N,9] -> Spalte 0 = Zentrum
        n_use = min(len(imgs), n_used)
        idx = rng.permutation(len(imgs))[:n_use]
        for i in idx:
            patch = imgs[i, 36:64, 36:64].astype(numpy.float32) / 255.0
            xs.append(patch)
            ys.append(int(labels[i, 0]))
            bs.append((0, 0, 28, 28, 0, 0, 28, 28))  # Box = ganze 28x28
    return xs, ys, bs


def _normalize_boxes(items, n):
    """items: Liste von (gx0,gy0,gx1,gy1, wx0,wy0,ww,wh) -> [N,4] normalisiert."""
    out = numpy.zeros((n, 4), dtype=numpy.float32)
    for i, (gx0, gy0, gx1, gy1, wx0, wy0, ww, wh) in enumerate(items):
        cx = (gx0 + gx1) / 2 - wx0
        cy = (gy0 + gy1) / 2 - wy0
        bw = gx1 - gx0
        bh = gy1 - gy0
        out[i] = (cx / ww, cy / wh, bw / ww, bh / wh)
    return out


def build_bnn_datasets(seed=42, base=None, max_pos_window=3, max_pos_tile=3,
                       neg_per_img_train=6, neg_per_img_test=6,
                       use_cluttered=True, n_cluttered=8000,
                       max_foreign=3):
    """
    Baut Train/Test-Tensordatasets (x, y_cls, y_box) fuer das BNN multitask.
    base: optionaler Scene-Datenbank-Ordner (z. B. SVHN-Szenen);
          None = Standard scene_dataset.
    """
    rng = numpy.random.default_rng(seed)
    tr = load_scene_split("train", base=base)
    te = load_scene_split("test", base=base)

    tr_wx, tr_wy, tr_wb = _window_samples(
        tr["images"], tr["boxes"], tr["labels"], rng, max_pos_window,
        max_foreign=max_foreign)
    tr_tx, tr_ty, tr_tb = _tile_samples(
        tr["images"], tr["boxes"], tr["labels"], rng, max_pos_tile)
    te_wx, te_wy, te_wb = _window_samples(
        te["images"], te["boxes"], te["labels"], rng, max_pos_window,
        max_foreign=max_foreign)
    te_tx, te_ty, te_tb = _tile_samples(
        te["images"], te["boxes"], te["labels"], rng, max_pos_tile)

    tr_nx, tr_ny, tr_nb = _negative_samples(tr["images"], tr["boxes"], rng,
                                            neg_per_img_train)
    te_nx, te_ny, te_nb = _negative_samples(te["images"], te["boxes"], rng,
                                            neg_per_img_test)

    au_x, au_y, au_b = (_clutter_crops(n_cluttered, rng) if use_cluttered
                        else ([], [], []))

    X_tr = numpy.stack(tr_wx + tr_tx + tr_nx + au_x).astype(numpy.float32)
    y_tr = numpy.asarray(tr_wy + tr_ty + tr_ny + au_y, dtype=numpy.int64)
    # Box-Targets: fuer BG-Klassen auf Dummy (maskiert) setzen
    box_tr = _normalize_boxes(tr_wb + tr_tb + tr_nb + au_b, len(y_tr))
    for i in range(len(y_tr)):
        if y_tr[i] == BG_CLASS:
            box_tr[i] = (0.5, 0.5, 0.0, 0.0)

    X_te = numpy.stack(te_wx + te_tx + te_nx).astype(numpy.float32)
    y_te = numpy.asarray(te_wy + te_ty + te_ny, dtype=numpy.int64)
    box_te = _normalize_boxes(te_wb + te_tb + te_nb, len(y_te))
    for i in range(len(y_te)):
        if y_te[i] == BG_CLASS:
            box_te[i] = (0.5, 0.5, 0.0, 0.0)

    X_tr = torch.from_numpy(X_tr).unsqueeze(1)
    y_tr = torch.from_numpy(y_tr)
    box_tr = torch.from_numpy(box_tr)
    X_te = torch.from_numpy(X_te).unsqueeze(1)
    y_te = torch.from_numpy(y_te)
    box_te = torch.from_numpy(box_te)
    return (TensorDataset(X_tr, y_tr, box_tr),
            TensorDataset(X_te, y_te, box_te))


if __name__ == "__main__":
    tr, te = build_bnn_datasets(max_pos_window=2, max_pos_tile=2,
                                neg_per_img_train=2, neg_per_img_test=2,
                                use_cluttered=True, n_cluttered=2000)
    import collections
    ytr = [int(tr[i][1]) for i in range(0, len(tr), 97)]
    yte = [int(te[i][1]) for i in range(0, len(te), 71)]
    print("Train:", len(tr), "Klassen:", collections.Counter(ytr))
    print("Test :", len(te), "Klassen:", collections.Counter(yte))
    print("Sample box (positiv):", tr[0][2].tolist())
    print("Sample box (hintergrund):", tr[-1][2].tolist())