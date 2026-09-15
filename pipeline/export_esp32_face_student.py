"""
ESP32-Export fuer den DISTILLIERTEN Studenten (Stufe 7, "Box = Fenster").

Der Student ist ein REINER Klassifikator (Gesicht/Hintergrund, 2 Klassen)
OHNE Box-Head (`box_head=False`) - die Bounding-Box kommt auf dem ESP32 wie
in Stufe 3+4 direkt aus dem adaptiven Fenster (Saliency), es gibt also KEIN
fc3 = fc3 mehr. Gegenueber dem Teacher entfaellt:
  - fc3 (4 Box-Neuronen, lfc3w/lfc3b/lfc3w8)
  - der riesige fc1-Kanal bleibt quantisiert, ist aber kleiner (c2=40 -> 1960
    statt 3920 Inputs), d.h. das f32-BNN-Blob wird ~4x kleiner.

Importiert einen von distill_mlp_face.py trainierten Checkpoint
(models/student_mlp_face.pt, state_dict zusaetzlich mit 'arch').

Blob-Layout face_bnn_student.bin (erkennbar in model.h):
  [c1w(24,3,3,3) c1b(24) c2w(40,24,3,3) c2b(40)
   fc1b(96) fc1s(96) fc1w8(96,1960) int8
   fc2w(2,96) fc2b(2)]
Saliency bleibt unveraendert (conv_mlp_saliency_face.pt -> face_saliency.bin).

Aufruf:
  python export_esp32_face_student.py
"""
import argparse
import os
import struct
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnn import BNN
from export_esp32_face import fold_bn, quantize_fc, write_blob, \
    export_saliency, F32, I8, IN_CH

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, "esp32", "models")
TEACHER_SALIENCY = "conv_mlp_saliency_face.pt"
STUDENT_CKPT = "student_mlp_face.pt"
SALIENCY_BLOB = "face_saliency.bin"
STUDENT_BLOB = "face_bnn_student.bin"


def export_student(model, path):
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saliency", default=TEACHER_SALIENCY)
    ap.add_argument("--student", default=STUDENT_CKPT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--saliency-only", action="store_true",
                    help="Nur face_saliency.bin neu schreiben (Saliency-Modell)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.saliency_only:
        ck = torch.load(os.path.join(MODELS, args.student),
                        map_location="cpu", weights_only=False)
        arch = ck.get("arch", {})
        print(f"Student-checkpoint arch: {arch}")
        student = BNN(n_class=2, box_head=False,
                      c1=arch.get("c1", 24), c2=arch.get("c2", 40),
                      hid=arch.get("hid", 96), in_ch=IN_CH)
        student.load_state_dict(ck["model"])
        student.eval()

        info = export_student(student,
                              os.path.join(args.out, STUDENT_BLOB))
        print("Student-Export:", info)
        print(f"  {STUDENT_BLOB}: "
              f"{os.path.getsize(os.path.join(args.out, STUDENT_BLOB))} B")

    # Saliency bleibt unveraendert: face_saliency.bin kommt aus dem Teacher-
    # Export (export_esp32_face.py). Wir schreiben es nur, falls es fehlt.
    saliency_path = os.path.join(args.out, SALIENCY_BLOB)
    if not os.path.isfile(saliency_path):
        print("Saliency-Blob fehlt - bitte zuerst export_esp32_face.py "
              "laufen lassen (face_saliency.bin wird dort erzeugt).")
    else:
        print(f"Saliency-Blob unveraendert: {SALIENCY_BLOB} "
              f"({os.path.getsize(saliency_path)} B)")

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"Gesamtgroesse: {total/1024:.1f} kB")
    print(f"Exportierte Modelle -> {args.out}")


if __name__ == "__main__":
    main()
