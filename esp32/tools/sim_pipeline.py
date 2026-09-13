"""Host-Simulator der ESP32-Gesichtserkennungs-Pipeline (reines NumPy).

Dieser Simulator ist die *ausfuehrbare Spezifikation* fuer die C-Firmware:
Die Firmware (esp32/firmware_ino/FaceDetectStream) rechnet Bit-fuer-Bit die
gleiche Kette. Wer die Firmware liest, kann jede Funktion 1:1 gegen dieses
Skript pruefen.

Kette (ganze Struktur, wie vom Nutzer gefordert):
  Kamera-Preprocessing : RGB565 -> RGB888 -> Center-Crop (quadratisch) ->
                         bilinear 128x128, /255
  1+2. Conv+ANFIS      : ConvStack(3->8->4) -> TileStats(mean/max/var) ->
                         Instanz-Norm ueber 64 Kacheln -> ANFIS(125 Regeln) ->
                         Sigmoid  => Saliency 8x8
  3.   Regionen        : Saliency >= REGION_THR -> Connected Components (4-Nachbar)
                         -> Crop-Fenster (Modus: hybrid2 | refocus | bbox)
  4.   BNN je Fenster  : Fenster -> bilinear 28x28 -> conv/pool/int8-FC1 ->
                         Klassen-Logits (softmax) + Box-Head (cx,cy,w,h)
  5.   Box in Bild     : (cx +- w/2)*W0, (cy +- h/2)*H0 + Fenster-Offset;
                         Gate + Hintergrund-Verwurf

Die Blobs (face_saliency.bin, face_bnn.bin) werden direkt geparst - es werden
KEINE .pt-Checkpoints benoetigt. Die Forward-Mathematik ist gegen
esp32/blobio/verify_blobs_final.py verifiziert.

Zweck: VOR dem Flashen auf einem echten Foto validieren und REGION_THR / gate /
Crop-Groessen / die FACE/BG-Klassenzuordnung kalibrieren. Das Skript druckt
Saliency-Statistik, die Regionenliste und je Fenster die rohen
Klassenwahrscheinlichkeiten + Box, und speichert ein annotiertes PNG.

Beispiele:
  python sim_pipeline.py --image foto.jpg
  python sim_pipeline.py --image foto.jpg --mode refocus --region-thr 0.5 --gate 0.6
  python sim_pipeline.py --image foto.jpg --face-class 0 --bg-class 1 --dump-saliency
"""
import argparse
import os
import sys

import numpy

# ----------------------------------------------------------------------------
# Pfade / Blob-Layout
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # .../dynamic-neural-network
MODELS_DIR = os.path.join(_ROOT, "esp32", "models")

F32 = 4
I8 = 1

# Grid-Geometrie (identisch zu data_common.py)
SCENE = 128
TILE = 16
GRID = 8          # SCENE // TILE
N_TILES = 64

# Standard-Schwellen (aus evaluate_pipeline.py) - per CLI ueberschreibbar
REGION_THR = 0.5
GATE = 0.6
BG_REJECT_P = 0.7   # cls==BG && p_max>=BG_REJECT_P -> verwerfen


class _Rd:
    """Sequenzieller Blob-Leser (little-endian, wie der Export sie schreibt)."""

    def __init__(self, b):
        self.b = b
        self.o = 0

    def arr(self, dt, shape):
        n = int(numpy.prod(shape))
        sz = n * dt
        d = numpy.float32 if dt == F32 else numpy.int8
        a = numpy.frombuffer(self.b[self.o:self.o + sz], dtype=d)
        self.o += sz
        return a.reshape(shape)


# ----------------------------------------------------------------------------
# Blob laden
# ----------------------------------------------------------------------------
def load_saliency_blob(path):
    """face_saliency.bin -> dict der Parameter (siehe verify_blobs_final.py)."""
    with open(path, "rb") as f:
        r = _Rd(f.read())
    P = {
        "c1w": r.arr(F32, (8, 3, 3, 3)), "c1b": r.arr(F32, (8,)),
        "c2w": r.arr(F32, (4, 8, 3, 3)), "c2b": r.arr(F32, (4,)),
        "c":   r.arr(F32, (3, 5)),
        "lgs": r.arr(F32, (3, 5)),
        "Pc":  r.arr(F32, (125, 4)),
    }
    P["sig"] = numpy.log1p(numpy.exp(P["lgs"]))    # softplus(log_sigma)
    if r.o != len(r.b):
        print(f"WARN: Saliency-Blob {r.o}/{len(r.b)} Bytes geparst")
    return P


def load_bnn_blob(path):
    """face_bnn.bin -> dict der Parameter (BN bereits in Conv gefaltet)."""
    with open(path, "rb") as f:
        r = _Rd(f.read())
    P = {
        "b1w": r.arr(F32, (40, 3, 3, 3)), "b1b": r.arr(F32, (40,)),
        "b2w": r.arr(F32, (80, 40, 3, 3)), "b2b": r.arr(F32, (80,)),
        "fc1b": r.arr(F32, (192,)), "fc1s": r.arr(F32, (192,)),
        "fc1w8": r.arr(I8, (192, 3920)),
        "fc2w": r.arr(F32, (2, 192)), "fc2b": r.arr(F32, (2,)),
        "fc3w": r.arr(F32, (4, 192)), "fc3b": r.arr(F32, (4,)),
    }
    if r.o != len(r.b):
        print(f"WARN: BNN-Blob {r.o}/{len(r.b)} Bytes geparst")
    return P


# ----------------------------------------------------------------------------
# NN-Primitive (identisch zur C-Firmware / verify_blobs_final.py)
# ----------------------------------------------------------------------------
def conv2d_same_relu(x, w, b):
    """x:[C,H,W] -> [O,H,W]; 3x3 Same-Pad, ReLU. float64-Akkumulation."""
    C, H, W = x.shape
    O = w.shape[0]
    xp = numpy.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = numpy.zeros((O, H, W), dtype=numpy.float32)
    for o in range(O):
        acc = numpy.full((H, W), float(b[o]), dtype=numpy.float64)
        for i in range(C):
            for dy in range(3):
                for dx in range(3):
                    acc += float(w[o, i, dy, dx]) * xp[i, dy:dy + H, dx:dx + W]
        out[o] = numpy.maximum(acc, 0.0)
    return out


def pool2(x):
    """[C,H,W] -> [C,H/2,W/2] 2x2-Maxpool."""
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def bilinear_resize(patch, size):
    """Zentriertes bilineares Resize (2D [H,W] oder 3D [H,W,C]) - wie data_common.crop."""
    h, w = patch.shape[:2]
    ys = numpy.linspace(0, h - 1, size)
    xs = numpy.linspace(0, w - 1, size)
    y0 = numpy.floor(ys).astype(numpy.int64)
    y1 = numpy.minimum(y0 + 1, h - 1)
    x0 = numpy.floor(xs).astype(numpy.int64)
    x1 = numpy.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    def _2d(pc, fy, fx, y0, y1, x0, x1):
        return (pc[y0, :][:, x0] * (1 - fy) * (1 - fx)
                + pc[y0, :][:, x1] * (1 - fy) * fx
                + pc[y1, :][:, x0] * fy * (1 - fx)
                + pc[y1, :][:, x1] * fy * fx)

    if patch.ndim == 2:
        v = _2d(patch, fy, fx, y0, y1, x0, x1)
    else:
        C = patch.shape[2]
        v = numpy.empty((size, size, C), dtype=numpy.float64)
        for c in range(C):
            v[:, :, c] = _2d(patch[:, :, c], fy, fx, y0, y1, x0, x1)
    return v.astype(numpy.float32)


# ----------------------------------------------------------------------------
# Stufe 1+2: Saliency
# ----------------------------------------------------------------------------
def saliency_forward(P, scene):
    """scene:[3,128,128] float 0..1 -> saliency [8,8] (Sigmoid-Wahrscheinlichkeit)."""
    f1 = conv2d_same_relu(scene, P["c1w"], P["c1b"])   # [8,128,128]
    f2 = conv2d_same_relu(f1, P["c2w"], P["c2b"])            # [4,128,128]

    # Tile-Statistik: je 16x16-Kachel ueber [4,16,16]=1024 Werte
    stats = numpy.zeros((GRID, GRID, 3), dtype=numpy.float64)
    for j in range(GRID):
        for i in range(GRID):
            blk = f2[:, j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
            stats[j, i, 0] = blk.mean()
            stats[j, i, 1] = blk.max()
            stats[j, i, 2] = blk.var()               # Populationsvarianz (ddof=0)

    # Instanz-Norm ueber die 64 Kacheln je Statistik (std mit ddof=1), clamp +-3
    for k in range(3):
        s = stats[:, :, k]
        stats[:, :, k] = numpy.clip((s - s.mean()) / (s.std(ddof=1) + 1e-5), -3, 3)

    # ANFIS je Kachel (125 Regeln, 5 MF pro Eingang)
    c, sig, Pc = P["c"], P["sig"], P["Pc"]
    sal = numpy.zeros((GRID, GRID), dtype=numpy.float64)
    for j in range(GRID):
        for i in range(GRID):
            xv = stats[j, i]
            w_ = numpy.ones(125)
            for r in range(125):
                m0, m1, m2 = (r // 25) % 5, (r // 5) % 5, r % 5
                mm = (m0, m1, m2)
                val = 1.0
                for k in range(3):
                    dk = xv[k] - c[k, mm[k]]
                    val *= numpy.exp(-(dk * dk) / (2.0 * sig[k, mm[k]] ** 2))
                w_[r] = val
            w_ = w_ / (w_.sum() + 1e-6)
            lin = Pc[:, 0] + Pc[:, 1] * xv[0] + Pc[:, 2] * xv[1] + Pc[:, 3] * xv[2]
            sal[j, i] = 1.0 / (1.0 + numpy.exp(-(w_ @ lin)))
    return sal


# ----------------------------------------------------------------------------
# Stufe 4: BNN (int8-FC1, deterministisch S=1, Dropout AUS)
# ----------------------------------------------------------------------------
def bnn_forward(P, crop28):
    """crop28:[3,28,28] float -> (probs[2], box[cx,cy,w,h] dekodiert)."""
    f1 = conv2d_same_relu(crop28, P["b1w"], P["b1b"])   # [40,28,28]
    p1 = pool2(f1)                                            # [40,14,14]
    f2 = conv2d_same_relu(p1, P["b2w"], P["b2b"])             # [80,14,14]
    p2 = pool2(f2)                                            # [80,7,7]
    flat = p2.reshape(-1).astype(numpy.float64)               # 3920 (C-Ordnung)

    # int8-FC1: h = relu( (flat @ W8^T) * fc1s + fc1b )
    h = numpy.maximum(
        (flat @ P["fc1w8"].astype(numpy.float64).T) * P["fc1s"] + P["fc1b"], 0.0)

    logits = P["fc2w"].astype(numpy.float64) @ h + P["fc2b"]
    e = numpy.exp(logits - logits.max())
    probs = e / e.sum()

    raw = P["fc3w"].astype(numpy.float64) @ h + P["fc3b"]
    box = numpy.array([
        1.0 / (1.0 + numpy.exp(-raw[0])),   # cx
        1.0 / (1.0 + numpy.exp(-raw[1])),   # cy
        numpy.exp(raw[2]),                  # w
        numpy.exp(raw[3]),                  # h
    ])
    return probs, box


# ----------------------------------------------------------------------------
# Stufe 3: Connected Components (4-Nachbar, reines NumPy - kein scipy)
# ----------------------------------------------------------------------------
def extract_regions(binary_mask, min_tiles=1):
    """[8,8]-Bool-Maske -> Regionen (dicts x0,y0,x1,y1,tiles), gross->klein.

    4-Nachbarschaft (oben/unten/links/rechts), identisch zu
    data_common.extract_regions (scipy plus-Struktur), aber per BFS.
    """
    H, W = binary_mask.shape
    lab = numpy.zeros((H, W), dtype=numpy.int32)
    regions = []
    comp = 0
    for sy in range(H):
        for sx in range(W):
            if not binary_mask[sy, sx] or lab[sy, sx]:
                continue
            comp += 1
            stack = [(sy, sx)]
            lab[sy, sx] = comp
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W \
                            and binary_mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = comp
                        stack.append((ny, nx))
            if len(cells) < min_tiles:
                continue
            ys = numpy.array([c[0] for c in cells])
            xs = numpy.array([c[1] for c in cells])
            regions.append({
                "x0": int(xs.min() * TILE), "y0": int(ys.min() * TILE),
                "x1": int((xs.max() + 1) * TILE), "y1": int((ys.max() + 1) * TILE),
                "tiles": int(len(cells)), "component": comp,
            })
    regions.sort(key=lambda r: -r["tiles"])
    return regions


# ----------------------------------------------------------------------------
# Crop-Strategien - VERBATIM aus pipeline/evaluate_pipeline.py (reines NumPy)
# Nur so bleibt die Fenster-Auswahl bit-identisch zur Referenz-Pipeline.
# ----------------------------------------------------------------------------
def _region_patches(img, regs, margin=2):
    patches = []
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
        x1 = min(128, x1 + margin); y1 = min(128, y1 + margin)
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches


def _refocus_patches(img, sal, regs, min_size=24, max_size=48):
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
    assign = numpy.zeros(len(pts), dtype=int)
    for _ in range(max_iter):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        nassign = numpy.argmin(d, axis=1)
        for c in range(k):
            m = nassign == c
            if m.sum() == 0:
                if seeds is not None:
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


def _saliency_peaks(sal, mask=None, thr=REGION_THR, win=3):
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


def _dedup_squares(sal, squares, iou_thr=0.6, imp_thr=REGION_THR):
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


def _patches_from_squares(img, squares):
    patches = []
    for cx, cy, half, _ in squares:
        x0 = int(numpy.clip(int(numpy.floor(cx - half)), 0, 128))
        y0 = int(numpy.clip(int(numpy.floor(cy - half)), 0, 128))
        x1 = int(numpy.clip(int(numpy.ceil(cx + half)), 0, 128))
        y1 = int(numpy.clip(int(numpy.ceil(cy + half)), 0, 128))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches


def _hybrid2_patches(img, sal, regs, big_extent=32, min_size=24, max_size=48,
                     peak_win=28, dedup_iou=0.75, peak_dedup_iou=0.85,
                     rng=None, imp_thr=REGION_THR):
    patches = []
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        extent = max(x1 - x0, y1 - y0)
        if extent <= big_extent:
            patches.extend(_refocus_patches(img, sal, [r], min_size, max_size))
            continue
        i0, j0 = x0 // 16, y0 // 16
        i1, j1 = (x1 - 1) // 16, (y1 - 1) // 16
        mask = numpy.zeros((8, 8), dtype=bool)
        mask[j0:j1 + 1, i0:i1 + 1] = sal[j0:j1 + 1, i0:i1 + 1] > 0
        peaks = _saliency_peaks(sal, mask=mask, thr=imp_thr)
        if len(peaks) == 0:
            patches.extend(_refocus_patches(img, sal, [r], min_size, max_size))
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
        squares = _dedup_squares(sal, squares, iou_thr=dedup_iou, imp_thr=imp_thr)
        for py, px in peaks:
            squares.append((px * 16.0 + 8.0, py * 16.0 + 8.0, peak_win / 2.0, -1.0))
        squares = _dedup_squares(sal, squares, iou_thr=peak_dedup_iou, imp_thr=imp_thr)
        patches.extend(_patches_from_squares(img, squares))
    return patches


def patches_for(img, sal, regs, mode, min_size, max_size, rng):
    if mode == "bbox":
        return _region_patches(img, regs)
    if mode == "refocus":
        return _refocus_patches(img, sal, regs, min_size, max_size)
    return _hybrid2_patches(img, sal, regs, min_size=min_size,
                            max_size=max_size, rng=rng)


# ----------------------------------------------------------------------------
# Kamera-Preprocessing (spiegelt die Firmware)
# ----------------------------------------------------------------------------
def load_rgb_scene(path, center_crop=True):
    """Bild -> [128,128,3] float 0..1 RGB, optional Center-Crop quadratisch.

    Spiegelt die Firmware: RGB565-Capture -> RGB888 -> zentraler quadratischer
    Ausschnitt -> bilinear 128x128 -> /255.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = numpy.load(path).astype(numpy.float64)
        if arr.ndim == 2:                       # Graustufen -> 3 ident. Kanaele
            arr = numpy.stack([arr] * 3, axis=2)
        if arr.max() > 1.5:
            arr = arr / 255.0
        rgb = arr[..., :3]
    else:
        try:
            from PIL import Image
        except ImportError:
            sys.exit("Pillow fehlt: 'pip install pillow' oder .npy-Bild nutzen.")
        im = Image.open(path).convert("RGB")
        rgb = numpy.asarray(im, dtype=numpy.float64) / 255.0

    if center_crop:
        h, w = rgb.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        rgb = rgb[y0:y0 + s, x0:x0 + s]
    return bilinear_resize(rgb, SCENE).astype(numpy.float32)


# ----------------------------------------------------------------------------
# Pipeline + Diagnose
# ----------------------------------------------------------------------------
def run(args):
    salP = load_saliency_blob(os.path.join(MODELS_DIR, "face_saliency.bin"))
    bnnP = load_bnn_blob(os.path.join(MODELS_DIR, "face_bnn.bin"))
    scene = load_rgb_scene(args.image, center_crop=not args.no_center_crop)
    scene_c = numpy.transpose(scene, (2, 0, 1))      # [3,128,128] fuer Conv
    rng = numpy.random.default_rng(args.seed)

    # ---- Saliency ----
    sal = saliency_forward(salP, scene_c)
    print(f"\n=== Saliency 8x8 (Schwelle REGION_THR={args.region_thr}) ===")
    print(f"  min={sal.min():.3f} max={sal.max():.3f} mean={sal.mean():.3f} "
          f"aktive Kacheln>=thr: {(sal >= args.region_thr).sum()}/64")
    if args.dump_saliency:
        for j in range(GRID):
            print("  " + " ".join(f"{sal[j, i]:.2f}" for i in range(GRID)))

    mask = sal >= args.region_thr
    regs = extract_regions(mask)
    print(f"\n=== Regionen (Connected Components, {len(regs)}) ===")
    for k, r in enumerate(regs):
        print(f"  R{k}: tiles={r['tiles']:2d} "
              f"bbox=({r['x0']},{r['y0']})-({r['x1']},{r['y1']})")

    patches = patches_for(scene, sal, regs, args.mode,
                          args.min_size, args.max_size, rng)
    print(f"\n=== Fenster (Modus={args.mode}, {len(patches)}) -> BNN ===")
    print(f"  FACE_CLASS={args.face_class}  BG_CLASS={args.bg_class}  gate={args.gate}")

    dets = []
    for pi, (patch, ox, oy) in enumerate(patches):
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        H0, W0 = patch.shape[0], patch.shape[1]
        crop28_l = bilinear_resize(patch, 28)             # [28,28,3]
        crop28 = numpy.transpose(crop28_l, (2, 0, 1))     # [3,28,28]
        probs, box = bnn_forward(bnnP, crop28)
        cls = int(numpy.argmax(probs))
        p_max = float(probs.max())
        cx, cy, w, h = box
        bx0 = (cx - w / 2) * W0 + ox
        by0 = (cy - h / 2) * H0 + oy
        bx1 = (cx + w / 2) * W0 + ox
        by1 = (cy + h / 2) * H0 + oy
        pf = float(probs[args.face_class])
        keep = "KEEP"
        if cls == args.bg_class and p_max >= args.bg_reject_p:
            keep = "drop(bg)"
        elif p_max < args.gate:
            keep = "drop(gate)"
        print(f"  W{pi}: win=({ox},{oy}) {W0}x{H0}  probs="
              f"[{probs[0]:.3f},{probs[1]:.3f}]  cls={cls} p_face={pf:.3f}  "
              f"box=({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f})  -> {keep}")
        if keep == "KEEP":
            dets.append({"box": (bx0, by0, bx1, by1), "cls": cls,
                         "conf": p_max, "p_face": pf})

    faces = [d for d in dets if d["cls"] == args.face_class]
    print(f"\n=== Ergebnis: {len(dets)} Detektion(en), davon {len(faces)} Gesicht(er) ===")
    for d in faces:
        x0, y0, x1, y1 = d["box"]
        print(f"  FACE p={d['conf']:.3f}  ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")

    if args.out:
        _save_annotated(scene, faces, args.out)
        print(f"\nAnnotiertes Bild -> {args.out}")

    print("\nHinweis: Stimmen p_face/Boxen nicht, --face-class/--bg-class tauschen "
          "oder --region-thr/--gate/--min-size/--max-size kalibrieren. Genau diese "
          "Werte gehen dann in esp32/firmware_ino/FaceDetectStream/config.h.")


def _save_annotated(scene, faces, out):
    """RGB-Szene + gruene Boxen speichern (PIL)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow fehlt - kein PNG gespeichert.")
        return
    rgb = (numpy.clip(scene, 0, 1) * 255).astype(numpy.uint8)
    pim = Image.fromarray(rgb, "RGB")
    dr = ImageDraw.Draw(pim)
    for d in faces:
        x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
        dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        dr.text((x0 + 1, max(0, y0 - 10)), f"{d['conf']:.2f}", fill=(0, 255, 0))
    pim.save(out)


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Host-Simulator der ESP32-Gesichtserkennungs-Pipeline (NumPy).")
    ap.add_argument("--image", required=True, help="Eingabebild (jpg/png/npy)")
    ap.add_argument("--mode", default="hybrid2",
                    choices=["hybrid2", "refocus", "bbox"],
                    help="Fenster-Strategie (Referenz-Default: hybrid2)")
    ap.add_argument("--region-thr", type=float, default=REGION_THR)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--bg-reject-p", type=float, default=BG_REJECT_P)
    ap.add_argument("--face-class", type=int, default=0,
                    help="Klassindex Gesicht (Annahme 0; per Sim kalibrieren)")
    ap.add_argument("--bg-class", type=int, default=1,
                    help="Klassindex Hintergrund (Annahme 1)")
    ap.add_argument("--min-size", type=int, default=24)
    ap.add_argument("--max-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-center-crop", action="store_true",
                    help="Bild NICHT quadratisch zuschneiden (direkt resizen)")
    ap.add_argument("--dump-saliency", action="store_true",
                    help="8x8-Saliency-Matrix ausgeben")
    ap.add_argument("--out", default=None, help="Pfad fuer annotiertes PNG")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
