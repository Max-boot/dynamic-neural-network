"""Blob-Paar-Verifikation face_saliency.bin + face_bnn.bin -> numpy vs torch.

Layout laut export_esp32_face.py:
  face_saliency.bin: c1w(8,3,3,3) c1b(8) c2w(4,8,3,3) c2b(4)
                     fc1w(16,12) fc1b(16) fc2w(1,16) fc2b(1)
  face_bnn.bin:      c1w(40,3,3,3) c1b(40) c2w(80,40,3,3) c2b(80)
                     fc1b(192) fc1s(192) fc1w8(192,3920) int8
                     fc2w(2,192) fc2b(2) fc3w(4,192) fc3b(4)
"""
import os
import sys

import numpy
import torch

# Pfade repo-relativ (Skript liegt in esp32/blobio/ -> zwei Ebenen bis Repo-Wurzel)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
from stage12 import ConvMLPSaliency, to_model_input
from bnn import BNN

MODELS = os.path.join(_ROOT, "pipeline", "models")
OUT = os.path.join(_ROOT, "esp32", "models")

F32 = 4
I8S = 1


class Rd:
    def __init__(self, b):
        self.b = b
        self.o = 0

    def arr(self, dt, shape):
        nf = int(numpy.prod(shape))
        sz = nf * dt
        d = numpy.float32 if dt == F32 else numpy.int8
        a = numpy.frombuffer(self.b[self.o:self.o + sz], dtype=d)
        self.o += sz
        return a.reshape(shape)


def conv2d_same(x, w, b):
    """x:[B,C,H,W] -> out:[B,O,H,W]; Same-Pad, ReLU."""
    B, C, H, W = x.shape
    O = w.shape[0]
    xp = numpy.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = numpy.zeros((B, O, H, W), dtype=numpy.float32)
    for o in range(O):
        acc = numpy.full((H, W), b[o], dtype=numpy.float64)
        for i in range(C):
            for dy in range(3):
                for dx in range(3):
                    acc += float(w[o, i, dy, dx]) * xp[0, i, dy:dy + H, dx:dx + W]
        out[:, o] = numpy.maximum(acc, 0.0)
    return out


def pool2(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def decode_box(raw):
    return numpy.array([1.0 / (1.0 + numpy.exp(-raw[0])),
                        1.0 / (1.0 + numpy.exp(-raw[1])),
                        numpy.exp(raw[2]), numpy.exp(raw[3])])


def fold_bn(w, b, bn):
    g = bn.weight.detach().numpy()
    s = 1.0 / numpy.sqrt(bn.running_var.numpy() + bn.eps)
    ww = w.detach().numpy() * (g * s)[:, None, None, None]
    bb = (b.detach().numpy() - bn.running_mean.numpy()) * (g * s) \
        + bn.bias.detach().numpy()
    return ww.astype(numpy.float32), bb.astype(numpy.float32)


def main():
    # ---------- Torch-Referenz laden ----------
    sal = ConvMLPSaliency(in_ch=3)
    sal.load_state_dict(torch.load(os.path.join(MODELS, "conv_mlp_saliency_face.pt"),
                                   map_location="cpu",
                                   weights_only=False)["model"])
    sal.eval()
    bnn = BNN(n_class=2, box_head=True, c1=40, c2=80, hid=192, in_ch=3)
    bnn.load_state_dict(torch.load(os.path.join(MODELS, "bnn_mc_box_face.pt"),
                                   map_location="cpu",
                                   weights_only=False)["model"])
    bnn.eval()

    # ---------- Saliency-Blob ----------
    with open(os.path.join(OUT, "face_saliency.bin"), "rb") as f:
        sb = f.read()
    r = Rd(sb)
    c1w = r.arr(F32, (8, 3, 3, 3)); c1b = r.arr(F32, (8,))
    c2w = r.arr(F32, (4, 8, 3, 3)); c2b = r.arr(F32, (4,))
    fc1w = r.arr(F32, (16, 12)); fc1b = r.arr(F32, (16,))
    fc2w = r.arr(F32, (1, 16)); fc2b = r.arr(F32, (1,))
    print(f"Saliency-Blob geparst: {r.o}/{len(sb)} Bytes")

    rng = numpy.random.default_rng(7)
    maxd_sal = 0.0
    t_sal_h = None
    for t in range(8):
        scene = rng.random((128, 128, 3)).astype(numpy.float32)
        with torch.no_grad():
            t_sal_h = torch.sigmoid(
                sal(torch.from_numpy(to_model_input(scene[None])))).numpy().reshape(8, 8)
        f1 = conv2d_same(to_model_input(scene[None]), c1w, c1b)
        f2 = conv2d_same(f1, c2w, c2b)
        # per-Kanal Tile-Stats -> [8,8,12], Reihenfolge c*3+{mean,max,var}
        stats = numpy.zeros((8, 8, 12))
        for j in range(8):
            for i in range(8):
                for cc in range(4):
                    blk = f2[0, cc, j * 16:(j + 1) * 16,
                             i * 16:(i + 1) * 16].astype(numpy.float64)
                    stats[j, i, cc * 3 + 0] = blk.mean()
                    stats[j, i, cc * 3 + 1] = blk.max()
                    stats[j, i, cc * 3 + 2] = blk.var()
        for k in range(12):
            s = stats[:, :, k]
            stats[:, :, k] = numpy.clip(
                (s - s.mean()) / (s.std(ddof=1) + 1e-5), -3, 3)
        n_sal = numpy.zeros((8, 8))
        for j in range(8):
            for i in range(8):
                xv = stats[j, i]                              # [12]
                h = numpy.maximum(fc1w @ xv + fc1b, 0.0)      # [16]
                y = float(fc2w[0] @ h + fc2b[0])              # Skalar
                n_sal[j, i] = 1.0 / (1.0 + numpy.exp(-y))
        maxd_sal = max(maxd_sal, numpy.abs(n_sal - t_sal_h).max())
    print(f"Saliency: numpy vs torch max|diff| = {maxd_sal:.2e}")

    # ---------- BNN-Blob ----------
    with open(os.path.join(OUT, "face_bnn.bin"), "rb") as f:
        bb = f.read()
    r2 = Rd(bb)
    b1w = r2.arr(F32, (40, 3, 3, 3)); b1b = r2.arr(F32, (40,))
    b2w = r2.arr(F32, (80, 40, 3, 3)); b2b = r2.arr(F32, (80,))
    fc1b = r2.arr(F32, (192,)); fc1s = r2.arr(F32, (192,))
    fc1w8 = r2.arr(I8S, (192, 3920))
    fc2w = r2.arr(F32, (2, 192)); fc2b = r2.arr(F32, (2,))
    fc3w = r2.arr(F32, (4, 192)); fc3b = r2.arr(F32, (4,))
    print(f"BNN-Blob geparst: {r2.o}/{len(bb)} Bytes")

    maxd1 = maxd2 = 0.0
    hits = 0
    fc1w_ref = bnn.fc1.weight.detach().numpy().astype(numpy.float64)
    # Symmetrische int8-Quantisierung: Betrags-Maximum je Zeile + Saettigung.
    # (Frueher wurde das vorzeichenbehaftete max verwendet -> int8-wrap bei
    #  Zeilen mit negativem Spitzenwert. Siehe export_esp32_face.py.)
    amax = numpy.max(numpy.abs(fc1w_ref), axis=1, keepdims=True)
    iscale = numpy.where(amax > 0.0, amax / 127.0, 1.0)
    w8_ref = numpy.clip(numpy.rint(fc1w_ref / iscale), -127, 127).astype(numpy.int8)
    int8_exact = numpy.array_equal(fc1w8, w8_ref)
    maxd_q = numpy.abs(fc1w8.astype(numpy.float64) -
                       w8_ref.astype(numpy.float64)).max() if not int8_exact else 0.0
    print(f"FC1-int8 vs round(w/fc1s): exakt={int8_exact} max|qdiff|={maxd_q:.2e} "
          f"scale-range [{fc1s.min():.2e},{fc1s.max():.2e}]")
    for t in range(16):
        crop = rng.random((28, 28, 3)).astype(numpy.float32)
        with torch.no_grad():
            # Deterministische Referenz: eval()-Forward (kein MC-Dropout),
            # identisch zu dem, was der ESP32 in float ausfuehrt.
            t_logits, t_boxr = bnn(
                torch.from_numpy(to_model_input(crop[None])))
            t_logits = t_logits.numpy()[0]
            el = numpy.exp(t_logits - t_logits.max())
            t_logits = el / el.sum()
            t_box = decode_box(t_boxr.numpy()[0])
        f1 = conv2d_same(to_model_input(crop[None]), b1w, b1b)
        p1 = pool2(f1)
        f2 = conv2d_same(p1, b2w, b2b)
        p2 = pool2(f2)
        flat = p2.reshape(-1)
        h = numpy.maximum(
            (flat.astype(numpy.float64) @ fc1w8.astype(numpy.float64).T)
            * fc1s + fc1b, 0.0)
        n_logits = fc2w.astype(numpy.float64) @ h + fc2b
        el = numpy.exp(n_logits - n_logits.max())
        n_mu = el / el.sum()
        raw = fc3w.astype(numpy.float64) @ h + fc3b
        n_box = numpy.array([raw[0], raw[1], raw[2], raw[3]])
        n_box[0] = 1.0 / (1.0 + numpy.exp(-raw[0]))
        n_box[1] = 1.0 / (1.0 + numpy.exp(-raw[1]))
        n_box[2] = numpy.exp(raw[2])
        n_box[3] = numpy.exp(raw[3])
        maxd1 = max(maxd1, numpy.abs(n_mu - t_logits).max())
        maxd2 = max(maxd2, numpy.abs(n_box - t_box).max())
        hits += int(numpy.argmax(n_logits) == int(t_logits.argmax()))
    print(f"BNN: max|diff| class-probs={maxd1:.3f} box={maxd2:.3f} "
          f"argmax-hits {hits}/16")

    ok = maxd_sal < 1e-3 and int8_exact and hits == 16
    print("RESULT:", "OK" if ok else "FEHL; siehe max|diff|/-hits/int8-exact")


    ok = maxd_sal < 1e-3 and maxd1 < 0.15 and maxd2 < 0.15 and hits == 16
    print("RESULT:", "OK" if ok else "FEHL; siehe max|diff|/-hits")


if __name__ == "__main__":
    main()
