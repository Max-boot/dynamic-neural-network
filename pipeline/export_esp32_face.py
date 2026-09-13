"""
Modell-Export fuer ESP32: Gesichtspipeline (Stage12-Saliency + BNN).

- Saliency (conv_mlp_saliency_face.pt): ConvStack(3->8->4) + MLP(12->16->1)
- BNN    (bnn_mc_box_face.pt):            Conv1(3ch)->BN1->Pool->Conv2->BN2->Pool
                                           -> FC3920x192 -> klassen(2) + box(4)

Export-Format (Binaries, Layout wird vom ESP32-Code als bekannt angenommen):
  face_saliency.bin   [c1w(8,3,3,3), c1b(8), c2w(4,8,3,3), c2b(4),
                        fc1w(16,12), fc1b(16), fc2w(1,16), fc2b(1)]
  face_bnn.bin        [c1w(40,3,3,3), c1b(40), c2w(80,40,3,3), c2b(80),
                        fc1b(192), fc1_scale(192), fc1w8(192,3920) int8,
                        fc2w(2,192), fc2b(2), fc3w(4,192), fc3b(4)]

BN wird in Conv gefaltet (gamma/sqrt(var+eps)). FC1 per-Ausgabekanal symmetrisch
int8 (scale = max|W|/127); Inferenz: y = (sum_j w8_ij * x_j) * scale_i + b_i.
"""
import os
import struct
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnn import BNN
from stage12 import ConvMLPSaliency

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "esp32", "models")

F32 = 4
I8 = 1


def fold_bn(conv_w, conv_b, bn):
    g = bn.weight.detach().numpy()
    b = bn.bias.detach().numpy()
    m = bn.running_mean.numpy()
    v = bn.running_var.numpy()
    s = 1.0 / numpy.sqrt(v + bn.eps)
    w = conv_w.detach().numpy() * (g * s)[:, None, None, None]
    bb = (conv_b.detach().numpy() - m) * (g * s) + b
    return w.astype(numpy.float32), bb.astype(numpy.float32)


def quantize_fc(w, b):
    w = w.detach().numpy()
    b = b.detach().numpy()
    scale = numpy.max(numpy.abs(w), axis=1) / 127.0
    scale = numpy.maximum(scale, 1e-12)
    w8 = numpy.round(w / scale[:, None]).astype(numpy.int8)
    return w8, scale.astype(numpy.float32), b.astype(numpy.float32)


def write_blob(path, chunks):
    content = b""
    for dt, arr in chunks:
        a = numpy.asarray(arr).reshape(-1)
        if dt == F32:
            content += a.astype(numpy.float32).tobytes()
        else:
            content += a.astype(numpy.int8).tobytes()
    with open(path, "wb") as f:
        f.write(content)
    print(f"  {os.path.basename(path)}: {len(content)} Bytes")


def export_saliency(model, path):
    conv1 = model.conv_stack.conv1
    conv2 = model.conv_stack.conv2
    fc1w = model.mlp.fc1.weight.detach().numpy().astype(numpy.float32)   # (16,12)
    fc1b = model.mlp.fc1.bias.detach().numpy().astype(numpy.float32)     # (16,)
    fc2w = model.mlp.fc2.weight.detach().numpy().astype(numpy.float32)   # (1,16)
    fc2b = model.mlp.fc2.bias.detach().numpy().astype(numpy.float32)     # (1,)
    write_blob(path, [
        (F32, conv1.weight.detach()), (F32, conv1.bias.detach()),
        (F32, conv2.weight.detach()), (F32, conv2.bias.detach()),
        (F32, fc1w), (F32, fc1b), (F32, fc2w), (F32, fc2b),
    ])
    return {"c1w": tuple(conv1.weight.shape), "c2w": tuple(conv2.weight.shape),
            "fc1w": fc1w.shape, "fc2w": fc2w.shape}


def export_bnn(model, path):
    c1w, c1b = fold_bn(model.conv1.weight, model.conv1.bias, model.bn1)
    c2w, c2b = fold_bn(model.conv2.weight, model.conv2.bias, model.bn2)
    fc1w8, fc1s, fc1b = quantize_fc(model.fc1.weight, model.fc1.bias)
    fc2w = model.fc2.weight.detach().numpy().astype(numpy.float32)
    fc2b = model.fc2.bias.detach().numpy().astype(numpy.float32)
    fc3w = model.fc3.weight.detach().numpy().astype(numpy.float32)
    fc3b = model.fc3.bias.detach().numpy().astype(numpy.float32)
    write_blob(path, [
        (F32, c1w), (F32, c1b),
        (F32, c2w), (F32, c2b),
        (F32, fc1b), (F32, fc1s),
        (I8, fc1w8),
        (F32, fc2w), (F32, fc2b),
        (F32, fc3w), (F32, fc3b),
    ])
    return {"c1w": c1w.shape, "c2w": c2w.shape,
            "fc1w8": fc1w8.shape, "fc1_scale_max": float(fc1s.max()),
            "fc1_scale_min": float(fc1s.min())}


IN_CH = 3


def main():
    os.makedirs(OUT, exist_ok=True)
    saliency = ConvMLPSaliency(in_ch=IN_CH)
    saliency.load_state_dict(torch.load(
        os.path.join(MODELS, "conv_mlp_saliency_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    info_s = export_saliency(saliency, os.path.join(OUT, "face_saliency.bin"))
    print("Saliency-Export:", info_s)

    bnn = BNN(n_class=2, box_head=True, c1=40, c2=80, hid=192, in_ch=IN_CH)
    bnn.load_state_dict(torch.load(
        os.path.join(MODELS, "bnn_mc_box_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    info_b = export_bnn(bnn, os.path.join(OUT, "face_bnn.bin"))
    print("BNN-Export:", info_b)

    total = sum(os.path.getsize(os.path.join(OUT, f))
                for f in os.listdir(OUT))
    print(f"Gesamtgroesse: {total/1024:.1f} kB")
    print(f"Exportierte Modelle -> {OUT}")


if __name__ == "__main__":
    main()