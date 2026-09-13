"""export_esp32_face.py  -  Torch-Checkpoints -> ESP32-Blobs (face_*.bin).

Schreibt die zwei Blobs, die die Firmware (esp32/firmware_ino/FaceDetectStream)
und sim_pipeline.py einlesen. Rekonstruiert aus der Spezifikation in
esp32/blobio/verify_blobs_final.py + verify_bnn_quant_final.py.

Layout (Byte-genau, wie von den Verify-Skripten geparst):
  face_saliency.bin: c1w(8,3,3,3) c1b(8) c2w(4,8,3,3) c2b(4)
                     fc1w(16,12) fc1b(16) fc2w(1,16) fc2b(1)         -> 2964 B
  face_bnn.bin:      c1w(40,3,3,3) c1b(40) c2w(80,40,3,3) c2b(80)
                     fc1b(192) fc1s(192) fc1w8(192,3920) int8
                     fc2w(2,192) fc2b(2) fc3w(4,192) fc3b(4)         -> 878808 B
  Alles float32 in C-Reihenfolge, ausser fc1w8 (int8).

WICHTIG - der behobene Bug:
  Die urspruengliche FC1-int8-Quantisierung benutzte das *vorzeichenbehaftete*
  Zeilen-Maximum als Skala:

      scale = w.max(axis=1) / 127          # FALSCH
      w8    = rint(w / scale).astype(int8) # laeuft ueber -> int8-wrap

  Bei jeder Zeile, deren betragsgroesstes Gewicht negativ ist (bei zentrierten
  Gewichten ~die Haelfte der 192 Zeilen), wird w/scale < -127; astype(int8)
  wrappt (z.B. -180 -> +76). Das verfaelscht FC1 und damit Klassen- und
  Box-Kopf -> unzuverlaessige Face/Background-Trennung.

  Symmetrische int8-Quantisierung MUSS das Betrags-Maximum verwenden und saettigen:

      amax  = max(|w|, axis=1)
      scale = amax / 127
      w8    = clip(round(w / scale), -127, 127).astype(int8)

  Der Forward auf dem Geraet ist skalen-generisch
  (h = relu(flat @ w8.T * fc1s + fc1b)); eine mit dieser Skala geschriebene
  fc1s macht die Dequantisierung exakt -> KEINE Firmware-Aenderung noetig.

Voraussetzung: die Face-Checkpoints in pipeline/models/:
  conv_mlp_saliency_face.pt     ("model" -> ConvMLPSaliency state_dict)
  bnn_mc_box_face.pt            ("model" -> BNN(n_class=2,...) state_dict)
Diese .pt liegen NICHT im Repo; ohne sie koennen die Blobs nicht neu erzeugt
werden (der alte Exporter und die .pt lebten ausserhalb dieses Repos).

Aufruf:
    cd esp32/blobio
    python export_esp32_face.py            # liest ../../pipeline/models, schreibt ../models
    python export_esp32_face.py --check    # zusaetzlich Selbst-Check der Dequant-Fehler
"""
import argparse
import os
import sys

import numpy
import torch

# Pfade repo-relativ (Skript liegt in esp32/blobio/ -> zwei Ebenen bis Repo-Wurzel)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
from stage12 import ConvMLPSaliency          # noqa: E402
from bnn import BNN                             # noqa: E402

MODELS = os.path.join(_ROOT, "pipeline", "models")
OUT = os.path.join(_ROOT, "esp32", "models")

SALIENCY_CKPT = "conv_mlp_saliency_face.pt"
BNN_CKPT = "bnn_mc_box_face.pt"
SALIENCY_BIN = "face_saliency.bin"
BNN_BIN = "face_bnn.bin"

# Sanity: exakte Blob-Groessen, die Firmware/Skripte erwarten.
SALIENCY_BYTES = 2964
BNN_BYTES = 878808
IN_CH = 3


# ---------------------------------------------------------------------------
def _load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _f32(t):
    """torch/np -> C-contiguous float32 ndarray."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return numpy.ascontiguousarray(t, dtype=numpy.float32)


def fold_conv_bn(conv, bn):
    """Conv+BatchNorm2d in eine Conv (w,b) falten. Identisch zu verify_*.fold_bn."""
    g = bn.weight.detach().numpy()
    s = 1.0 / numpy.sqrt(bn.running_var.numpy() + bn.eps)
    w = conv.weight.detach().numpy()
    b = (conv.bias.detach().numpy() if conv.bias is not None
         else numpy.zeros(w.shape[0], dtype=numpy.float64))
    ww = w * (g * s)[:, None, None, None]
    bb = (b - bn.running_mean.numpy()) * (g * s) + bn.bias.detach().numpy()
    return _f32(ww), _f32(bb)


def quant_fc1(weight):
    """Symmetrische per-Zeile int8-Quantisierung (Betrags-Max, gesaettigt).

    weight: (rows, cols) float. Rueckgabe (w8 int8, scale float32 per Zeile).
    """
    w = weight.astype(numpy.float64)
    amax = numpy.max(numpy.abs(w), axis=1, keepdims=True)      # (rows,1)
    scale = numpy.where(amax > 0.0, amax / 127.0, 1.0)         # keine /0
    w8 = numpy.clip(numpy.rint(w / scale), -127, 127).astype(numpy.int8)
    return w8, scale.reshape(-1).astype(numpy.float32)


# ---------------------------------------------------------------------------
def export_saliency(out_path):
    sal = ConvMLPSaliency(in_ch=IN_CH)
    sal.load_state_dict(_load_state(os.path.join(MODELS, SALIENCY_CKPT)))
    sal.eval()

    parts = [
        _f32(sal.conv_stack.conv1.weight),   # c1w (8,3,3,3)
        _f32(sal.conv_stack.conv1.bias),     # c1b (8,)
        _f32(sal.conv_stack.conv2.weight),   # c2w (4,8,3,3)
        _f32(sal.conv_stack.conv2.bias),     # c2b (4,)
        _f32(sal.mlp.fc1.weight),            # fc1w (16,12)
        _f32(sal.mlp.fc1.bias),              # fc1b (16,)
        _f32(sal.mlp.fc2.weight),            # fc2w (1,16)
        _f32(sal.mlp.fc2.bias),              # fc2b (1,)
    ]
    blob = b"".join(p.tobytes() for p in parts)
    _write(out_path, blob, SALIENCY_BYTES, "saliency")
    return sal


def export_bnn(out_path):
    bnn = BNN(n_class=2, box_head=True, c1=40, c2=80, hid=192, in_ch=IN_CH)
    bnn.load_state_dict(_load_state(os.path.join(MODELS, BNN_CKPT)))
    bnn.eval()

    b1w, b1b = fold_conv_bn(bnn.conv1, bnn.bn1)   # (40,1,3,3),(40,)
    b2w, b2b = fold_conv_bn(bnn.conv2, bnn.bn2)   # (80,40,3,3),(80,)
    fc1w = bnn.fc1.weight.detach().numpy()        # (192,3920)
    fc1w8, fc1s = quant_fc1(fc1w)
    fc1b = _f32(bnn.fc1.bias)                     # (192,)

    parts_f32_pre = [b1w, b1b, b2w, b2b, fc1b, fc1s]   # bis vor fc1w8
    parts_f32_post = [
        _f32(bnn.fc2.weight), _f32(bnn.fc2.bias),      # (2,192),(2,)
        _f32(bnn.fc3.weight), _f32(bnn.fc3.bias),      # (4,192),(4,)
    ]
    blob = (b"".join(p.tobytes() for p in parts_f32_pre)
            + numpy.ascontiguousarray(fc1w8, dtype=numpy.int8).tobytes()
            + b"".join(p.tobytes() for p in parts_f32_post))
    _write(out_path, blob, BNN_BYTES, "bnn")
    return bnn, fc1w, fc1w8, fc1s


def _write(path, blob, expect, tag):
    if len(blob) != expect:
        raise SystemExit(
            f"[{tag}] Blob-Groesse {len(blob)} B != erwartet {expect} B - "
            f"Layout/Hyperparameter pruefen.")
    with open(path, "wb") as f:
        f.write(blob)
    print(f"[{tag}] geschrieben: {path}  ({len(blob)} B, OK)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Face-Detection-Blobs fuer ESP32 exportieren.")
    ap.add_argument("--out-dir", default=OUT, help="Zielordner fuer die .bin (Default esp32/models)")
    ap.add_argument("--check", action="store_true",
                    help="Dequant-Fehler von FC1 numerisch berichten")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name in (SALIENCY_CKPT, BNN_CKPT):
        p = os.path.join(MODELS, name)
        if not os.path.isfile(p):
            raise SystemExit(
                f"Checkpoint fehlt: {p}\n"
                f"Die Face-Checkpoints liegen nicht im Repo. Ohne sie koennen die "
                f"Blobs nicht erzeugt werden - Checkpoints wiederherstellen/neu trainieren.")

    export_saliency(os.path.join(args.out_dir, SALIENCY_BIN))
    _bnn, fc1w, fc1w8, fc1s = export_bnn(os.path.join(args.out_dir, BNN_BIN))

    if args.check:
        deq = fc1w8.astype(numpy.float64) * fc1s[:, None]
        err = numpy.abs(deq - fc1w.astype(numpy.float64))
        rng = numpy.abs(fc1w).max(axis=1)
        print(f"[check] FC1 dequant max|err|={err.max():.3e}  "
              f"mittlerer rel-Fehler={numpy.mean(err.max(axis=1) / (rng + 1e-12)):.3e}  "
              f"(erwartet ~ scale/2, << 1)")
        over = numpy.sum((numpy.rint(fc1w / fc1s[:, None]) > 127) |
                         (numpy.rint(fc1w / fc1s[:, None]) < -127))
        print(f"[check] Werte ausserhalb [-127,127] vor clip: {int(over)} "
              f"(mit abs-max erwartet 0)")

    print("\nFertig. Blobs jetzt per esp32/tools/upload_model.py auf den ESP32 laden "
          "und einmal neu starten.")


if __name__ == "__main__":
    main()
