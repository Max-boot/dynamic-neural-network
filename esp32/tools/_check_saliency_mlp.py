"""_check_saliency_mlp.py - Torch-freier Selbsttest des MLP-Saliency-Kopfs.

Prueft OHNE PyTorch/WIDER-Daten/Checkpoint drei Dinge an einem synthetischen
2964-B-Blob (c1w(8,3,3,3) c1b(8) c2w(4,8,3,3) c2b(4) fc1w(16,12) fc1b(16)
fc2w(1,16) fc2b(1)):

  1. Layout    : load_saliency_blob parst byte-genau (r.o == len; jede Teilmatrix
                 hat die erwartete Form + Sentinel-Wert -> Offsets stimmen).
  2. Analytik  : bei Null-Conv (Gewichte 0, nur Bias) ist f2 pro Kanal konstant
                 -> alle 12 Features nach Instanz-Norm exakt 0 -> die Saliency
                 jeder der 64 Kacheln ist der von Hand gerechnete Skalar
                 sigmoid(fc2b + relu(fc1b) @ fc2w).
  3. Voller Pfad: eine unabhaengige NumPy-Referenz (eigene Stats/Norm/MLP-Schleifen,
                 Feature-Reihenfolge c*3+s) gegen saliency_forward auf Zufalls-
                 szenen (max|diff| < 1e-6).

Damit sind Layout + Feature-Reihenfolge + Instanz-Norm + MLP-Mathematik der Sim
belegt - genau die Kette, die nn.cpp nn_saliency 1:1 spiegelt.

Aufruf:  python esp32/tools/_check_saliency_mlp.py
"""
import os
import sys
import tempfile

import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sim_pipeline import (load_saliency_blob, saliency_forward,
                          conv2d_same_relu, GRID, TILE, SCENE)

F32 = numpy.float32


def _blob(c1w, c1b, c2w, c2b, fc1w, fc1b, fc2w, fc2b):
    """Serialisiert die 8 Arrays in Export-Reihenfolge zu float32-Bytes."""
    parts = [c1w, c1b, c2w, c2b, fc1w, fc1b, fc2w, fc2b]
    return b"".join(numpy.asarray(a, dtype=F32).tobytes() for a in parts)


def _ref_forward(P, scene):
    """Unabhaengige Referenz des MLP-Saliency-Forward (eigene Schleifen)."""
    f1 = conv2d_same_relu(scene, P["c1w"], P["c1b"])
    f2 = conv2d_same_relu(f1, P["c2w"], P["c2b"])
    feat = numpy.zeros((GRID * GRID, 12), dtype=numpy.float64)
    for j in range(GRID):
        for i in range(GRID):
            t = j * GRID + i
            for c in range(4):
                blk = f2[c, j * TILE:(j + 1) * TILE,
                         i * TILE:(i + 1) * TILE].astype(numpy.float64)
                feat[t, c * 3 + 0] = blk.mean()
                feat[t, c * 3 + 1] = blk.max()
                feat[t, c * 3 + 2] = blk.var()          # Populationsvarianz (ddof=0)
    for k in range(12):
        col = feat[:, k]
        feat[:, k] = numpy.clip(
            (col - col.mean()) / (col.std(ddof=1) + 1e-5), -3.0, 3.0)
    fc1w, fc1b = P["fc1w"], P["fc1b"]
    fc2w, fc2b = P["fc2w"], P["fc2b"]
    out = numpy.zeros((GRID, GRID), dtype=numpy.float64)
    for j in range(GRID):
        for i in range(GRID):
            xv = feat[j * GRID + i]
            h = numpy.maximum(fc1w @ xv + fc1b, 0.0)
            y = float(fc2w[0] @ h + fc2b[0])
            out[j, i] = 1.0 / (1.0 + numpy.exp(-y))
    return out


def test_layout(tmp):
    """Sentinel-Blob: jede Teilmatrix konstant mit eigenem Wert."""
    sent = {
        "c1w": (numpy.full((8, 3, 3, 3), 1.0), 1.0),
        "c1b": (numpy.full((8,), 2.0), 2.0),
        "c2w": (numpy.full((4, 8, 3, 3), 3.0), 3.0),
        "c2b": (numpy.full((4,), 4.0), 4.0),
        "fc1w": (numpy.full((16, 12), 5.0), 5.0),
        "fc1b": (numpy.full((16,), 6.0), 6.0),
        "fc2w": (numpy.full((1, 16), 7.0), 7.0),
        "fc2b": (numpy.full((1,), 8.0), 8.0),
    }
    order = ["c1w", "c1b", "c2w", "c2b", "fc1w", "fc1b", "fc2w", "fc2b"]
    with open(tmp, "wb") as f:
        f.write(_blob(*[sent[k][0] for k in order]))
    P = load_saliency_blob(tmp)
    ok = True
    for k in order:
        want_arr, want_val = sent[k]
        got = P[k]
        if got.shape != want_arr.shape or not numpy.allclose(got, want_val):
            print(f"  [layout] FEHL {k}: shape {got.shape} vs {want_arr.shape}, "
                  f"val~{float(numpy.asarray(got).flat[0]):.3f} vs {want_val}")
            ok = False
    print(f"  [layout] {'OK' if ok else 'FEHL'} "
          f"(alle 8 Arrays formtreu + korrekte Offsets)")
    return ok


def test_analytic(tmp):
    """Null-Conv -> Features 0 -> Saliency = sigmoid(fc2b + relu(fc1b)@fc2w)."""
    rng = numpy.random.default_rng(1)
    c1w = numpy.zeros((8, 3, 3, 3)); c1b = rng.standard_normal(8)
    c2w = numpy.zeros((4, 8, 3, 3)); c2b = rng.standard_normal(4)
    fc1w = rng.standard_normal((16, 12))                 # irrelevant (x=0)
    fc1b = rng.standard_normal(16)
    fc2w = rng.standard_normal((1, 16))
    fc2b = rng.standard_normal(1)
    with open(tmp, "wb") as f:
        f.write(_blob(c1w, c1b, c2w, c2b, fc1w, fc1b, fc2w, fc2b))
    P = load_saliency_blob(tmp)
    scene = rng.random((3, SCENE, SCENE)).astype(numpy.float32)
    sal = saliency_forward(P, scene)
    h = numpy.maximum(fc1b, 0.0)                          # relu(fc1b), da x=0
    y = float(fc2w[0] @ h + fc2b[0])
    expected = 1.0 / (1.0 + numpy.exp(-y))
    d = float(numpy.abs(sal - expected).max())
    ok = d < 1e-6
    print(f"  [analytic] {'OK' if ok else 'FEHL'} von-Hand={expected:.6f} "
          f"max|diff ueber 64 Kacheln|={d:.2e}")
    return ok


def test_full(tmp):
    """Zufaellige Gewichte + Szenen: saliency_forward vs unabh. Referenz."""
    rng = numpy.random.default_rng(7)
    maxd = 0.0
    for _ in range(5):
        c1w = rng.standard_normal((8, 3, 3, 3)) * 0.1
        c1b = rng.standard_normal(8) * 0.1
        c2w = rng.standard_normal((4, 8, 3, 3)) * 0.1
        c2b = rng.standard_normal(4) * 0.1
        fc1w = rng.standard_normal((16, 12)) * 0.5
        fc1b = rng.standard_normal(16) * 0.5
        fc2w = rng.standard_normal((1, 16)) * 0.5
        fc2b = rng.standard_normal(1) * 0.5
        with open(tmp, "wb") as f:
            f.write(_blob(c1w, c1b, c2w, c2b, fc1w, fc1b, fc2w, fc2b))
        P = load_saliency_blob(tmp)
        scene = rng.random((3, SCENE, SCENE)).astype(numpy.float32)
        got = saliency_forward(P, scene)
        ref = _ref_forward(P, scene)
        maxd = max(maxd, float(numpy.abs(got - ref).max()))
    ok = maxd < 1e-6
    print(f"  [full] {'OK' if ok else 'FEHL'} sim vs Referenz "
          f"max|diff| = {maxd:.2e}")
    return ok


def main():
    fd, tmp = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        r = [test_layout(tmp), test_analytic(tmp), test_full(tmp)]
    finally:
        os.remove(tmp)
    ok = all(r)
    print("RESULT:", "OK" if ok else "FEHL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
