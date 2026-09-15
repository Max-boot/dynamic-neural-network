"""
ESP32-Export fuer den DISTILLIERTEN Studenten (Stufe 7, "Box = Fenster").

Der Student ist ein REINER Klassifikator (Gesicht/Hintergrund, 2 Klassen)
OHNE Box-Head (`box_head=False`). Die Bounding-Box kommt auf dem ESP32 wie in
Stufe 3+4 direkt aus dem adaptiven Fenster (Saliency) - es gibt also KEIN
fc3 (= Box-Output) mehr nur noch:
  c1w c1b c2w c2b fc1b fc1s fc1w8(int8) fc2w fc2b

Saliency-Blob bleibt unveraendert (face_saliency.bin aus Stufe 3-Weights,
conv_mlp_saliency_face.pt).

Blob face_bnn_student.bin (erkennbar in model.h):
  [c1w(24,3,3,3) c1b(24) c2w(40,24,3,3) c2b(40)
   fc1b(96) fc1s(96) fc1w8(96,1960) int8
   fc2w(2,96) fc2b(2)]
Saliency bleibt unveraendert (conv_mlp_saliency_face.pt -> face_saliency.bin).
Gegenueber export_esp32_face.py entfaellt fc3 (kein Box-Head am Studenten).

Aufruf:
  python export_esp32_face_student.py --student models/student_mlp_face.pt
"""
import argparse
import os
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnn import BNN
from export_esp32_face import fold_bn, quantize_fc, write_blob, \
    export_saliency, F32, I8, IN_CH

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "esp32", "models")
SALIENCY_CKPT = "conv_mlp_saliency_face.pt"
STUDENT_CKPT = "student_mlp_face.pt"
SALIENCY_BLOB = "face_saliency.bin"
STUDENT_BLOB = "face_bnn_student.bin"


def export_bnn(model, path):
    c1w, c1b = fold_bn(model.conv1.weight, model.conv1.bias, model.bn1)
    c2w, c2b = fold_bn(model.conv2.weight, model.conv2.bias, model.bn2)
    fc1w8, fc1s, fc1b = quantize_fc(model.fc1.weight, model.fc1.bias)
    fc2w = model.fc2.weight.detach().numpy().astype(numpy.float32)
    fc2b = model.fc2.bias.detach().numpy().astype(numpy.float32)
    write_blob(path, [
        (F32, c1w), (F32, c1b),
        (F32, c2w), (F32, c2b),
        (F32, fc1b), (F32, fc1s),
        (I8, fc1w8),
        (F32, fc2w), (F32, fc2b),
    ])
    return {"c1w": tuple(c1w.shape), "c2w": tuple(c2w.shape),
            "fc1w8": tuple(fc1w8.shape),
            "fc1_scale_max": float(fc1s.max())}


def main():
    ck = torch.load(os.path.join(MODELS, "conv_mlp_saliency_face.pt"),
                    map_location="cpu", weights_only=False)
    saliency = ConvMLPSaliency(in_ch=IN_CH)
    saliency.load_state_dict(ck["model"])
    saliency.eval()
    export_saliency(saliency, os.path.join(OUT, SALIENCY_BLOB))

    student = BNN(n_class=2, box_head=False, c1=24, c2=40, hid=96, in_ch=3)
    student.load_state_dict(torch.load(
        os.path.join(MODELS, "student_mlp_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    student.eval()
    export_bnn(student, os.path.join(OUT, STUDENT_BLOB))

    total = sum(os.path.getsize(os.path.join(OUT, f))
                for f in os.listdir(OUT))
    print(f"Gesamtgroesse: {total/1024:.1f} kB")
    print(f"Exportierte Modelle -> {OUT}")
    print("Hinweis: Auf dem ESP32 ist die Box = adaptives Fenster (kein fc3).")


if __name__ == "__main__":
    main()
