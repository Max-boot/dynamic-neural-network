"""
Archiv: Vorbereitete Crop-Strategien (Stand nach dem Strategie-Vergleich).

Dieses Modul dokumentiert die verworfenen Vorgehensweisen, die in der
endgueltigen Pipeline NICHT mehr verwendet werden. Die aktive Strategie ist
'hybrid2' (siehe pipeline/evaluate_pipeline.py: _hybrid2_patches); ihr
Vorgaenger 'hybrid' ist dort weiterhin als Modus enthalten.

Bewertung siehe CROP_STRATEGIEN.md. Kernaussage:

  - bbox    (rohe Tile-Bounding-Box einer Saliency-Region)
  - refocus (Quadrat um den saliency-gewichteten Schwerpunkt einer Region)
  - kmeans  (Lloyd-k-means ueber das ganze 8x8-Saliency-Grid, Fenster je
             Cluster -> viele redundante Fenster, viele False Positives)
  - hybrid  (k-means NUR in grossen verschmolzenen Regionen + Refokus fuer
             kleine; Quadrat-Ableitung mit harter Fremd-Exklusion)

hybrid gewann die erste Messung (Recall 0.0489, img_tp 0.340). Die
Weiterentwicklung hybrid2 ersetzt die k-Schätzung (Masse/1.5 -> Saliency-
Peaks), repariert leere Cluster und zentriert die Fenster ohne harte
Fremd-Exklusion plus Waisen-Absicherung -> GT-Zentrum-Coverage 58.3 % ->
97.8 %, Recall 0.0555, cls_acc 0.883 (1000-Bilder-Messung).

Der Code hier ist ein natuerlichsprachlicher Extrakt des damaligen Stands;
die Referenz-Implementationen sind die Funktionen, die zum Messzeitpunkt in
pipeline/evaluate_pipeline.py existierten.
"""
import numpy

TILE = 16
REGION_THR = 0.5
MIN_SIZE, MAX_SIZE = 24, 48


# ---------------------------------------------------------------- bbox ----
def bbox_patches(img, regs, margin=2):
    """Rohe Tile-Bounding-Box der Region (mit +-margin px Rand).

    Problem: die Ziffer steht meist nicht im Fensterzentrum (Bounding-Box
    umfasst nur die grob gefuellten Kacheln der Saliency >= 0.5). Dadurch
    liegt die Ziffer oft am Rand, der BNN-Box-Head lokalisierte falsch.
    """
    patches = []
    for r in regs:
        x0 = max(0, r["x0"] - margin)
        y0 = max(0, r["y0"] - margin)
        x1 = min(128, r["x1"] + margin)
        y1 = min(128, r["y1"] + margin)
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches


# ------------------------------------------------------------- refocus ----
def refocus_patches(img, sal, regs, min_size=MIN_SIZE, max_size=MAX_SIZE):
    """Quadratisches Fenster um den saliency-gewichteten Schwerpunkt.

    Groesse = Regions-Ausdehnung, geclampt auf [min_size, max_size] (deckt
    das BNN-Window 24..48px ab). Verbessert die Zentrierung deutlich, aber:
      * bei verschmolzenen Mehrfach-Ziffern (eine grosse Region) sitzt der
        Schwerpunkt zwischen den Ziffern -> kein Fenster deckt eine Ziffer
        isoliert ab;
      * Recall bleibt deshalb maessig (0.0380 vs. hybrid 0.0489).
    """
    patches = []
    for r in regs:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        i0, j0 = x0 // TILE, y0 // TILE
        i1, j1 = (x1 - 1) // TILE, (y1 - 1) // TILE
        sub = sal[j0:j1 + 1, i0:i1 + 1]
        wsum = float(sub.sum())
        if wsum <= 0:
            continue
        ys_, xs_ = numpy.indices(sub.shape)
        cy_t = float((sub * ys_).sum() / wsum)
        cx_t = float((sub * xs_).sum() / wsum)
        cx = x0 + (cx_t + 0.5) * TILE
        cy = y0 + (cy_t + 0.5) * TILE
        extent = max(x1 - x0, y1 - y0)
        w = int(min(max(extent, min_size), max_size))
        xc0 = int(numpy.clip(cx - w / 2, 0, 128 - w))
        yc0 = int(numpy.clip(cy - w / 2, 0, 128 - w))
        patches.append((img[yc0:yc0 + w, xc0:xc0 + w], xc0, yc0))
    return patches


# -------------------------------------------------------------- kmeans ----
def kmeans_patches(img, sal, mass_per_digit=2.0, max_k=12, cent_cls=None,
                   rng=None, min_size=MIN_SIZE, max_size=MAX_SIZE):
    """Lloyd-k-means direkt auf dem ganzen 8x8-Saliency-Grid.

    k = round(Saliency-Masse / Tiles-pro-Ziffer), clamp [1, max_k].
    Fenster pro Cluster: grosses Raster-Quadrat, das alle wichtigen eigenen
    Tiles enthaelt und keine wichtigen Punkte fremder Cluster schneidet.

    Problem: k-means ohne CC-Kontext legt k Fenster quer ueber das Grid;
    ein Fenster deckt haeufig mehrere Ziffern / Hintergrundanteile, das BNN
    bekommt keine klar isolierte Ziffer -> viele sicher klassifizierte FPs
    (Precision ~0.25 auch nach Tuning). Mehrere Cluster-Split-/Merge-/NMS-
    Verbesserungen halfen (Precision 0.212->0.254, cls_acc 0.808->0.870,
    fwd 65.9->49.9), aber kmeans blieb dominiert (recall < 0.022).
    """
    mass = float(sal.sum())
    if mass <= 0:
        return []
    k = int(min(max_k, max(1, round(mass / mass_per_digit))))
    centers, _extents, assign, xs, ys = _lloyd(sal, k, rng=rng)
    squares = _squares(sal, centers, assign, xs, ys,
                       min_size=min_size, max_size=max_size)
    return _make_patches(img, squares)


# ----------------------------- gemeinsame k-means-Bausteine --------------
def _lloyd(sal, k, rng=None, max_iter=30):
    """Lloyd mit k-means++-Init auf Kachel-Zentren (Pixel-Koordinaten)."""
    if rng is None:
        rng = numpy.random.default_rng(42)
    mask = sal > 0
    ys, xs = numpy.nonzero(mask)
    if len(ys) == 0:
        return [], [], numpy.zeros(0, dtype=int), xs, ys
    w = sal[ys, xs]
    pts = numpy.stack([xs * 16.0 + 8.0, ys * 16.0 + 8.0], axis=1)
    k = int(max(1, min(k, len(pts))))
    centers = numpy.zeros((k, 2))
    centers[0] = pts[int(numpy.argmax(w))]
    for c in range(1, k):
        mind = numpy.full(len(pts), numpy.inf)
        for prev in range(c):
            d2 = ((pts - centers[prev][None, :]) ** 2).sum(1)
            mind = numpy.minimum(mind, d2)
        prob = w * (mind + 1e-9)
        if prob.sum() <= 0:
            centers[c] = pts[c % len(pts)]
        else:
            centers[c] = pts[rng.choice(len(pts), p=prob / prob.sum())]
    assign = numpy.zeros(len(pts), dtype=int)
    for _ in range(max_iter):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        na = numpy.argmin(d, axis=1)
        for c in range(k):
            m = na == c
            if m.sum() == 0:
                centers[c] = pts[numpy.argmax(
                    ((pts - centers[0][None, :]) ** 2).sum(1))]
            else:
                centers[c] = (pts[m] * w[m][:, None]).sum(0) / w[m].sum()
        if (na == assign).all():
            break
        assign = na
    return centers, [], assign, xs, ys


def _squares(sal, centers, assign, xs, ys, min_size=MIN_SIZE,
             max_size=MAX_SIZE, imp_thr=REGION_THR):
    """Groesstes grid-aligned Quadrat je Cluster: alle eigenen wichtigen
    Tiles enthalten, keine wichtigen Tiles fremder Cluster geschnitten."""
    squares = []
    for c in range(len(centers)):
        own = (assign == c) & (sal[xs, ys] >= imp_thr)
        if not own.any():
            continue
        ox0 = xs[own].min() * 16
        oy0 = ys[own].min() * 16
        ox1 = (xs[own].max() + 1) * 16
        oy1 = (ys[own].max() + 1) * 16
        w = max(ox1 - ox0, oy1 - oy0, min_size)
        w = int(min(w, max_size))
        cx = (ox0 + ox1) / 2
        cy = (oy0 + oy1) / 2
        squares.append((cx, cy, w / 2, c))
    return squares


def _make_patches(img, squares):
    patches = []
    for cx, cy, half, _ in squares:
        x0 = int(numpy.clip(numpy.floor(cx - half), 0, 128))
        y0 = int(numpy.clip(numpy.floor(cy - half), 0, 128))
        x1 = int(numpy.clip(numpy.ceil(cx + half), 0, 128))
        y1 = int(numpy.clip(numpy.ceil(cy + half), 0, 128))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patches.append((img[y0:y1, x0:x1], x0, y0))
    return patches