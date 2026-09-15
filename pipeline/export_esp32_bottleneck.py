"""
ESP32-Export fuer die Linear-Bottleneck-Saliency (Stage12-Experiment).

Ersetzt den MLP-Saliency-Kopf (conv_stack + TileStats + MLP) durch ein reines
Conv-Netz, das direkt die 8x8-Saliency-Logits ausgibt:

  stem : 3x3 s2 3->8    +BN+ReLU        [64]   (128->64)
  ir1  : 1x1  8->16 E   +BN+ReLU        [64]
         3x3-dw s2 16   +BN+ReLU        [32]
         1x1 16->8  P   +BN (linear)    [32]
  ir2  : 1x1  8->24 E   +BN+ReLU        [32]
         3x3-dw s2 24   +BN+ReLU        [16]
         1x1 24->12 P   +BN (linear)    [16]
  ir3  : 1x1 12->24 E   +BN+ReLU        [16]
         3x3-dw s2 24   +BN+ReLU        [8]
         1x1 24->12 P   +BN (linear)    [8]
  ir4  : 1x1 12->24 E   +BN+ReLU        [8]
         3x3-dw s1 24   +BN+ReLU        [8]
         1x1 24->8  P   +BN (linear)    [8]
  head : 1x1  8->1        (linear)      [8]  ->  flatten -> 64 Logits

Alle BN werden in die Conv gefaltet (gamma/sqrt(var+eps)). Ausgabe = Logits;
der ESP32 wendet danach die Sigmoid an. Blob-Layout face_saliency_bottleneck.bin
ist die einfache Sequenz der 14 Layer in Vorwaertsrichtung:
  [w1,b1] [w2,b2] ... [w13,b13] [w14]        (head hat keinen Bias)

Aufruf:
  python export_esp32_bottleneck.py
"""
import os
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage12 import ConvBottleneckSaliency, to_model_input

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, "esp32", "models")
CKPT = os.path.join(MODELS, "saliency_bottleneck_exp.pt")
BLOB = "face_saliency_bottleneck.bin"

F32 = 4


def fold_bn(conv_w, conv_b, bn):
    """BN in Conv falten. conv_biased = None (bias=False) wird als 0 behandelt."""
    g = bn.weight.detach().numpy()
    b = bn.bias.detach().numpy()
    m = bn.running_mean.numpy()
    v = bn.running_var.numpy()
    s = 1.0 / numpy.sqrt(v + bn.eps)
    w = conv_w.detach().numpy() * (g * s)[:, None, None, None]
    cb = (numpy.zeros_like(m) if conv_b is None
          else conv_b.detach().numpy())
    bb = (cb - m) * (g * s) + b
    return w.astype(numpy.float32), bb.astype(numpy.float32)


def write_blob(path, chunks):
    content = b"".join(numpy.asarray(a).reshape(-1).astype(numpy.float32).tobytes()
                       for dt, a in chunks if dt == F32)
    with open(path, "wb") as f:
        f.write(content)
    return len(content)


def layer_defs(model):
    """(w, b_after_fold, name) fuer alle 14 Conv-Layer in Vorwaertsrichtung."""
    conv = model.stem
    bn = model.stem

    def F(conv_l, bn_l, name, has_bias=False):
        w, b = fold_bn(conv_l.weight, conv_l.bias if has_bias else None, bn_l)
        return w, b, name

    def ir(block, name):
        m = block.conv
        e = F(m[0], m[1], name + ".expand")
        d = F(m[3], m[4], name + ".dw")     # depthwise: w (O,1,3,3)
        p = F(m[6], m[7], name + ".proj")
        return [e, d, p]

    layers = [F(conv[0], bn[1], "stem")]
    for i, bname in enumerate(["ir1", "ir2", "ir3", "ir4"]):
        layers += ir(getattr(model, bname), bname)
    layers.append((model.head.weight.detach().numpy().astype(numpy.float32),
                   None, "head"))
    return layers


STRIDE2 = {"stem", "ir1.dw", "ir2.dw", "ir3.dw"}   # ir4.dw ist s1


def numpy_forward(layers, scene):
    """Fp64-Referenz in C++-Tap-Reihenfolge (kein ReLU nach Project/Head)."""
    x = scene                       # [C,H,W] float64
    for w, b, name in layers:
        w = w.astype(numpy.float64)
        O, C, k, _ = w.shape
        H, W = x.shape[1], x.shape[2]
        is_dw = (C == 1)
        Cin = 1 if is_dw else C
        stride = 2 if name in STRIDE2 else 1
        relu = not (name.endswith(".proj") or name == "head")
        pad = 1 if k == 3 else 0
        OH = (H + 2 * pad - k) // stride + 1
        OW = (W + 2 * pad - k) // stride + 1
        out = numpy.zeros((O, OH, OW), dtype=numpy.float64)
        for o in range(O):
            bp = 0.0 if b is None else b[o]
            for y in range(OH):
                for qx in range(OW):
                    acc = float(bp)
                    for i in range(Cin):
                        ci = o if is_dw else i
                        for dy in range(k):
                            yy = y * stride + dy - pad
                            if yy < 0 or yy >= H:
                                continue
                            for dx in range(k):
                                xx = qx * stride + dx - pad
                                if xx < 0 or xx >= W:
                                    continue
                                acc += w[o, i, dy, dx] * x[ci, yy, xx]
                    out[o, y, qx] = max(acc, 0.0) if relu else acc
        x = out
    return x


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("Checkpoint-Metriken:", {k: round(float(v), 4)
          for k, v in ck.get("metrics", {}).items() if isinstance(v, float)})
    model = ConvBottleneckSaliency(in_ch=3)
    model.load_state_dict(ck["model"])
    model.eval()

    layers = layer_defs(model)
    n_w = sum(int(numpy.prod(w.shape)) for w, _, _ in layers)
    n_b = sum(int(numpy.prod(b.shape)) for _, b, _ in layers if b is not None)
    print(f"Convolution: {len(layers)} Layer, {n_w} Gewichte, {n_b} Biases, "
          f"{n_w + n_b} Floats = {(n_w + n_b) * 4} B")
    for w, b, name in layers:
        print(f"  {name:12s} w{w.shape} b{('None' if b is None else b.shape)}")

    # nn.cpp-Blob-Layout ist [w1,b1][w2,b2]...[w13,b13][w14] (head ohne Bias).
    chunks = []
    for w, b, name in layers:
        chunks.append((F32, w))
        if b is not None:
            chunks.append((F32, b))
    nbytes = write_blob(os.path.join(OUT, BLOB), chunks)
    print(f"{BLOB}: {nbytes} Bytes ({len(chunks)} Chunks)")

    # --- numpy-vs-torch Selbsttest (fp64, gleiche Tap-Reihenfolge) ----------
    rng = numpy.random.default_rng(42)
    maxd = 0.0
    for t in range(4):
        scene = rng.random((128, 128, 3)).astype(numpy.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(to_model_input(scene[None])))[0]
            ref = torch.sigmoid(logits).numpy().reshape(8, 8)
        n_out = numpy_forward(layers, to_model_input(scene[None])[0].astype(numpy.float64))
        n_sal = 1.0 / (1.0 + numpy.exp(-n_out.reshape(8, 8)))
        maxd = max(maxd, float(numpy.abs(n_sal - ref).max()))
    print(f"Selbsttest: numpy (gefaltet) vs torch max|diff| = {maxd:.2e}")
    assert maxd < 1e-4, "Folding-Fehler: numpy vs torch weicht zu stark ab"


if __name__ == "__main__":
    main()