"""Blob-Verifikation face_saliency_bottleneck.bin -> numpy (aus Datei) vs torch.

Layout laut export_esp32_bottleneck.py / model.cpp parse_bn_saliency():
  [w1,b1][w2,b2]...[w13,b13][w14]  (head ohne Bias), Sequenz:
  stem(8,3,3,3|8) ir1{exp(16,8,1,1|16) dw(16,1,3,3|16) proj(8,16,1,1|8)}
  ir2{exp(16,8,1,1|16) dw(16,1,3,3|16) proj(12,16,1,1|12)}
  ir3{exp(24,12,1,1|24) dw(24,1,3,3|24) proj(12,24,1,1|12)}
  ir4{exp(24,12,1,1|24) dw(24,1,3,3|24) proj(8,24,1,1|8)}
  head(1,8,1,1|kein Bias)
Gesamt: 2576 Gewichte + 208 Biases = 2784 floats = 11136 B (== SALIENCY_BOTTLENECK_BYTES).

Der Vorwaertslauf kommt aus export_esp32_bottleneck.numpy_forward (fp64,
C++-Tap-Reihenfolge) und wird mit dem Torch-Checkpoint verglichen; Toleranz
< 1e-3 (deckt zusaetzlich float32-Rauschen des ESP32 ab, wie bei der MLP-
Referenz). Aufruf: python verify_saliency_bottleneck_final.py
"""
import os
import sys

import numpy
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
from stage12 import ConvBottleneckSaliency, to_model_input
from export_esp32_bottleneck import numpy_forward

MODELS = os.path.join(_ROOT, "pipeline", "models")
OUT = os.path.join(_ROOT, "esp32", "models")
BLOB = os.path.join(OUT, "face_saliency_bottleneck.bin")
EXPECTED_BYTES = 11136

# (Anzahl Gewichte, Anzahl Biases) je 14 Layer - identisch zu model.cpp.
LAYERS = [
    (8 * 3 * 3 * 3, 8),       # stem
    (16 * 8, 16), (16 * 9, 16), (8 * 16, 8),    # ir1 e/d/p
    (16 * 8, 16), (16 * 9, 16), (12 * 16, 12),  # ir2
    (24 * 12, 24), (24 * 9, 24), (12 * 24, 12), # ir3
    (24 * 12, 24), (24 * 9, 24), (8 * 24, 8),   # ir4
    (1 * 8, 0),                                   # head (kein Bias)
]
NAMES = ["stem",
         "ir1.expand", "ir1.dw", "ir1.proj",
         "ir2.expand", "ir2.dw", "ir2.proj",
         "ir3.expand", "ir3.dw", "ir3.proj",
         "ir4.expand", "ir4.dw", "ir4.proj",
         "head"]

# 4D-Gewichtsshape je Layer (fuer numpy_forward, das (O,C,k,k) erwartet).
SHAPES = [(8, 3, 3, 3),
          (16, 8, 1, 1), (16, 1, 3, 3), (8, 16, 1, 1),
          (16, 8, 1, 1), (16, 1, 3, 3), (12, 16, 1, 1),
          (24, 12, 1, 1), (24, 1, 3, 3), (12, 24, 1, 1),
          (24, 12, 1, 1), (24, 1, 3, 3), (8, 24, 1, 1),
          (1, 8, 1, 1)]


def read_layers():
    data = numpy.fromfile(BLOB, dtype=numpy.float32)
    assert data.size * 4 == EXPECTED_BYTES, \
        f"Blob-Groesse {data.size * 4} B != {EXPECTED_BYTES}"
    o = 0
    layers = []
    for (nw, nb), shape, name in zip(LAYERS, SHAPES, NAMES):
        w = data[o:o + nw].reshape(shape); o += nw
        b = None
        if nb:
            b = data[o:o + nb]; o += nb
        layers.append((w, b, name))
    assert o == data.size, f"Offset {o} != {data.size} floats"
    return layers


def main():
    ck = torch.load(os.path.join(MODELS, "saliency_bottleneck_exp.pt"),
                    map_location="cpu", weights_only=False)
    model = ConvBottleneckSaliency(in_ch=3)
    model.load_state_dict(ck["model"])
    model.eval()

    layers = read_layers()
    print("Blob gelesen: 14 Layer (w,b)-Sequenz, Layout + Groesse ok")

    rng = numpy.random.default_rng(7)
    maxd = 0.0
    for t in range(8):
        scene = rng.random((128, 128, 3)).astype(numpy.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(to_model_input(scene[None])))[0]
            ref = torch.sigmoid(logits).numpy().reshape(8, 8)
        n_out = numpy_forward(
            layers, to_model_input(scene[None])[0].astype(numpy.float64))
        n_sal = 1.0 / (1.0 + numpy.exp(-n_out.reshape(8, 8)))
        maxd = max(maxd, float(numpy.abs(n_sal - ref).max()))
    print(f"numpy (aus Blob-Datei) vs torch max|diff| = {maxd:.2e}")
    ok = maxd < 1e-3
    print("RESULT:", "OK" if ok else "FEHL; siehe max|diff|")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())