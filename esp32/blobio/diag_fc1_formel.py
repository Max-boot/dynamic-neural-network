"""FC1-Blob: Welche int8-Rundungs-/Skalierungsformel erzeugt exakt den geschriebenen Blob?

Ziel: Für den ESP32-C-Forward deterministisch festlegen, wie FC1 dequantisiert wird.
Verglichen wird blob-fc1w8 (int8 aus face_bnn.bin) gegen Kandidaten aus Torch-fc1.weight.
"""
import os
import sys

import numpy
import torch

# Pfade repo-relativ (Skript liegt in esp32/blobio/ -> zwei Ebenen bis Repo-Wurzel)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
from bnn import BNN

F32 = 4
I8S = 1

BLOB = os.path.join(_ROOT, "esp32", "models", "face_bnn.bin")


class Rd:
    def __init__(self, b):
        self.b = b
        self.o = 0

    def arr(self, dt, shape):
        d = numpy.float32 if dt == F32 else numpy.int8
        n = int(numpy.prod(shape))
        a = numpy.frombuffer(self.b[self.o:self.o + n * dt], dtype=d)
        self.o += n * dt
        return a.reshape(shape)


def main():
    with open(BLOB, "rb") as f:
        bb = f.read()
    r = Rd(bb)
    r.arr(F32, (40, 3, 3, 3)); r.arr(F32, (40,))
    r.arr(F32, (80, 40, 3, 3)); r.arr(F32, (80,))
    fc1b = r.arr(F32, (192,)); fc1s = r.arr(F32, (192,))
    blob8 = r.arr(I8S, (192, 3920)).astype(numpy.int16)
    print(f"Blob-Parse: {r.o}/{len(bb)} (FC1-pos ok)")

    bnn = BNN(n_class=2, box_head=True, c1=40, c2=80, hid=192, in_ch=3)
    bnn.load_state_dict(torch.load(
        os.path.join(_ROOT, "pipeline", "models", "bnn_mc_box_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    bnn.eval()
    w = bnn.fc1.weight.detach().numpy().astype(numpy.float64)

    amax = numpy.max(numpy.abs(w), axis=1)
    forms = {
        "a: round-half-away max/127": (
            numpy.where(w / (amax / 127.0)[:, None] >= 0,
                        numpy.floor(w / (amax / 127.0)[:, None] + 0.5),
                        numpy.ceil(w / (amax / 127.0)[:, None] - 0.5)),
            numpy.maximum(amax / 127.0, 1e-12)),
        "b: rint half-even  max/127": (
            numpy.rint(w / (amax / 127.0)[:, None]),
            numpy.maximum(amax / 127.0, 1e-12)),
        "c: trunc  max/127": (
            numpy.trunc(w / (amax / 127.0)[:, None]),
            numpy.maximum(amax / 127.0, 1e-12)),
        "d: round-half-away max/255": (
            numpy.where(w / (amax / 255.0)[:, None] >= 0,
                        numpy.floor(w / (amax / 255.0)[:, None] + 0.5),
                        numpy.ceil(w / (amax / 255.0)[:, None] - 0.5)),
            numpy.maximum(amax / 255.0, 1e-12)),
    }
    print(f"Blob fc1s[0:4] = {fc1s[:4]}")
    for name, (cand, _sc) in forms.items():
        row_exact = numpy.sum(numpy.all(cand == blob8, axis=1))
        elem_exact = numpy.sum(cand == blob8)
        print(f"{name:35s} rows-exakt {row_exact}/192 elem-exakt "
              f"{elem_exact}/752640")

    cand_a = forms["a: round-half-away max/127"][0]
    diff = numpy.where(cand_a != blob8)
    print(f"\nMax-Diff-argsmax-Vergleich mit a[]:")
    if len(diff[0]):
        idx = numpy.unique(diff[0])[:6]
        print(f"  betroffene Zeilen: {idx.tolist()}")
        for i in idx:
            c = numpy.where(cand_a[i] != blob8[i])[0][:4]
            print(f"  Zeile {i}: pos={c.tolist()} blob={blob8[i, c].tolist()} "
                  f"cand={cand_a[i, c].tolist()}")
            m = numpy.argmax(numpy.abs(w[i]))
            print(f"    |w|max pos={m} w={w[i, m]:.4f} -> q blob="
                  f"{blob8[i, m]} cand={cand_a[i, m]}")

    exact_rows = numpy.all(forms["a: round-half-away max/127"][0] == blob8,
                           axis=1)
    ok = int(exact_rows.sum()) == 192
    print("\nRESULT:", "FC1-int8-Formel exakt bestätigt (a)" if ok else
          "FEHL; Formel-Kandidaten oben pruefen, skale-Diff in fc1b/fc1s? "
          "(Blob fc1s vs amax/127)")


if __name__ == "__main__":
    main()
