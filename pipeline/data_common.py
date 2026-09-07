"""
Gemeinsame Daten-/Label-Helfer fuer die Dynamic-NN-Pipeline (Stufen 1-5).

Stellt bereit:
  - Laden der Scene-Splits (train/test) + Ground-Truth (Ziffern-Boxen, Labels)
  - Tile-Grid-Logik: 128x128 Szene -> 8x8 Raster (16x16 Kacheln)
  - Tile-Ground-Truth: Kachel == positiv, wenn sie eine Ziffern-Box ueberlappt
  - Region-Crops fuer das BNN (Kachel / Region -> 28x28 Resize)
  - IoU / Matching-Helfer fuer die Detektions-Evaluation
"""
import os

import numpy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENE_SIZE = 128
TILE = 16
GRID = SCENE_SIZE // TILE  # 8
N_TILES = GRID * GRID      # 64


def load_scene_split(split: str = "train", base=None):
    """Laedt einen Scene-Split (.npz) und gibt dict mit Arrays zurueck."""
    if base is None:
        base = os.path.join(PROJECT_ROOT, "scene_dataset")
    path = os.path.join(base, f"scene_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = numpy.load(path)
    return {
        "images": d["images"].astype(numpy.float32) / 255.0,   # [N,128,128] 0..1
        "boxes": d["digit_boxes"].astype(numpy.float32),       # [N,K,4]
        "labels": d["digit_labels"].astype(numpy.int64),       # [N,K]
        "sizes": d["digit_sizes"].astype(numpy.int64),         # [N,K]
    }


def boxes_to_tiles(boxes):
    """
    Wandelt Ziffern-Boxen (x0,y0,x1,y1) in NxGRIDxGRID-Tile-Ground-Truth um.
    pos Kachel == 1, wenn Ueberlappung (Intersektion > 0) mit mind. einer Box.
    boxes: [N, K, 4] (padding -1)
    out  : [N, GRID, GRID] int64 (0/1)
    """
    N, K, _ = boxes.shape
    out = numpy.zeros((N, GRID, GRID), dtype=numpy.int64)
    for n in range(N):
        for k in range(K):
            x0, y0, x1, y1 = boxes[n, k]
            if x0 < 0:
                continue
            # Kachel-Indizes, die die Box beruehren
            i0 = int(max(0, x0 // TILE))
            i1 = int(min(GRID - 1, (x1 - 1) // TILE))
            j0 = int(max(0, y0 // TILE))
            j1 = int(min(GRID - 1, (y1 - 1) // TILE))
            out[n, j0:j1 + 1, i0:i1 + 1] = 1
    return out


def tile_coords(grid=GRID, tile=TILE):
    """Liefert je Kachel die (x0,y0,x1,y1) Koordinaten: [GRID, GRID, 4]."""
    coords = numpy.zeros((grid, grid, 4), dtype=numpy.int64)
    for j in range(grid):
        for i in range(grid):
            coords[j, i, 0] = i * tile
            coords[j, i, 1] = j * tile
            coords[j, i, 2] = (i + 1) * tile
            coords[j, i, 3] = (j + 1) * tile
    return coords


def crop(patch, size=28):
    """
    Zentriertes bilineares Resize eines 2D-Uint8/float32-Patches auf (size,size).
    Nutzt nur numpy (kein torch), damit es in der Datengenerierung funktioniert.
    """
    h, w = patch.shape
    ys = numpy.linspace(0, h - 1, size).astype(numpy.float64)
    xs = numpy.linspace(0, w - 1, size).astype(numpy.float64)
    y0 = numpy.floor(ys).astype(numpy.int64)
    y1 = numpy.minimum(y0 + 1, h - 1)
    x0 = numpy.floor(xs).astype(numpy.int64)
    x1 = numpy.minimum(x0 + 1, w - 1)
    fy = ys - y0
    fx = xs - x0
    v = patch[y0, :][:, x0] * (1 - fy)[:, None] * (1 - fx)[None, :] \
        + patch[y0, :][:, x1] * (1 - fy)[:, None] * fx[None, :] \
        + patch[y1, :][:, x0] * fy[:, None] * (1 - fx)[None, :] \
        + patch[y1, :][:, x1] * fy[:, None] * fx[None, :]
    return v.astype(numpy.float32) if v.dtype != numpy.uint8 else v


def extract_regions(binary_mask, min_tiles=1):
    """
    Connected Components (4-Nachbarschaft) auf [GRID,GRID]-Maske -> Regionen.
    Liefert Liste von dicts: {'x0','y0','x1','y1','component','tiles'}.
    """
    import scipy.ndimage as ndi
    structure = numpy.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=numpy.int8)
    lab, n = ndi.label(binary_mask, structure=structure)
    regions = []
    for c in range(1, n + 1):
        ys, xs = numpy.where(lab == c)
        if len(ys) < min_tiles:
            continue
        x0 = xs.min() * TILE
        y0 = ys.min() * TILE
        x1 = (xs.max() + 1) * TILE
        y1 = (ys.max() + 1) * TILE
        regions.append({
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "tiles": int(len(ys)), "component": int(c),
        })
    regions.sort(key=lambda r: -r["tiles"])
    return regions, lab


def boxes_iou(a, b):
    """IoU(Menge a, Menge b); a,b: (x0,y0,x1,y1)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0