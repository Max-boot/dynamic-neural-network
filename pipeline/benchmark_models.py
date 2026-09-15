"""
Laufzeit-Benchmark: unsere Saliency/Pipeline vs Referenzmodelle.

Misst auf DIESEM Rechner (CUDA oder CPU) ueber mehrere Iterationen die
durchschnittliche Inferenzzeit, Parameter und MACs. Absolutwerte sind
PC/Laufzeit-abhaengig, die VERHAELTNISSE sind das Vergleichskriterium.

Verglichen:
  1. ConvMLPSaliency        (aktuell deployt, MLP-Kopf 741 params)
  2. ConvBottleneckSaliency (Experiment-Gewinner, 2992 params)
  3. Pipeline komplett      (Bottleneck-Saliency + Student-BNN, CROP 28)
  4. MobileNetV2 (Google)   (torchvision, 128x128 Input -> Referenz)
  5. MobileNetV2 x0.25      (maximal geschrumpfte Google-Variante)

Qualitaet: eigene Metriken (AUROC/AP/recall + Pipeline precision/recall) werden
aus den Training-Ergebnissen geladen; Referenzmodelle sind ImageNet-Classifier
(dort ist MobilenetV2 top-1 acc. ~86.4%), d.h. die Qualitaet ist NICHT direkt
vergleichbar - sie dienen als Laufzeit-/Groessen-Referenz.
"""
import os
import sys
import time

import numpy
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage12 import ConvMLPSaliency, ConvBottleneckSaliency
from bnn import BNN

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

N_ITER = 300          # gemessene Iterationen
N_WARMUP = 30         # Warmup-Iterationen
IN_SIZE = 128
CROP = 28


def count_macs(model, size):
    """Zaehlt MACs ueber Conv/Linear-Hooks fuer Shape (1,3,size,size).
    Korrekt fuer Stride>1: Ausgabe-Spatial (out.shape[2:]) wird benutzt."""
    macs = [[0]]

    def hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            in_c = module.in_channels
            out_c = module.out_channels
            k = module.weight.shape[2] * module.weight.shape[3]
            o = out[0] if isinstance(out, (tuple, list)) else out
            hy, wx = o.shape[2], o.shape[3]
            macs[0][0] += in_c * out_c * k * hy * wx
        elif isinstance(module, nn.Linear):
            i = inp[0] if isinstance(inp, (tuple, list)) else inp
            macs[0][0] += int(numpy.prod(i.shape)) * module.out_features
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, size, size))
    for h in handles:
        h.remove()
    return macs[0][0]


def bench(model, size=IN_SIZE):
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    x = torch.rand(1, 3, size, size, device=dev)
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    ms = dt / N_ITER * 1000.0
    n_params = sum(p.numel() for p in model.parameters())
    try:
        macs = count_macs(model.cpu(), size)
    except Exception:
        macs = -1
    return {"ms": ms, "params": n_params, "macs": macs, "dev": dev}


def load_mlp():
    m = ConvMLPSaliency(in_ch=3)
    m.load_state_dict(torch.load(os.path.join(MODELS, "conv_mlp_saliency_face.pt"),
                                 map_location="cpu", weights_only=False)["model"])
    return m


def load_bn():
    m = ConvBottleneckSaliency(in_ch=3)
    m.load_state_dict(torch.load(os.path.join(MODELS, "saliency_bottleneck_exp.pt"),
                                 map_location="cpu", weights_only=False)["model"])
    return m


def load_pipeline():
    m = BNN(n_class=2, box_head=False, c1=24, c2=40, hid=96, in_ch=3)
    m.load_state_dict(torch.load(os.path.join(MODELS, "student_mlp_face.pt"),
                                 map_location="cpu", weights_only=False)["model"])
    return m


def main():
    print(f"Iterationen: {N_ITER}  Warmup: {N_WARMUP}")
    models = {
        "MLP-Saliency (deployt)":    (load_mlp, IN_SIZE),
        "Bottleneck-Saliency":       (load_bn, IN_SIZE),
        "BNN (Student, Crop28)":     (load_pipeline, CROP),
        "MobileNetV2 (Google, 1.0x)": (lambda: mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT).features, IN_SIZE),
        "MobileNetV2 (Google, 0.25x)": (lambda: mobilenet_v2(
            weights=None, width_mult=0.25).features, IN_SIZE),
    }

    rows = []
    for name, (factory, size) in models.items():
        try:
            r = bench(factory(), size)
            rows.append((name, r))
            print(f"  {name:<32} {r['ms']*1000:8.1f} us  "
                  f"params={r['params']:>8}  MACs={r['macs']/1e6:9.2f}M")
        except Exception as e:
            print(f"  {name:<32} FEHLER: {e}")

    # Pipeline-Schaetzung: Saliency 1x + BNN je Region (mean_fwd aus Eval).
    mean_fwd = 4.12
    t_sal = next(r["ms"] for n, r in rows if n == "Bottleneck-Saliency")
    t_bnn = next(r["ms"] for n, r in rows if n == "BNN (Student, Crop28)")
    mac_b = next(r["macs"] for n, r in rows if n == "Bottleneck-Saliency")
    p_b = next(r["params"] for n, r in rows if n == "Bottleneck-Saliency")
    p_bnn = next(r["params"] for n, r in rows if n == "BNN (Student, Crop28)")
    t_pipe = t_sal + mean_fwd * t_bnn
    rows.insert(3, ("Pipeline (~4x BNN/Frame)", {
        "ms": t_pipe, "params": p_b + p_bnn, "macs": mac_b + mean_fwd * 2_390_000,
        "dev": "est"}))
    print(f"  {'Pipeline (~4x BNN/Frame)':<32} {t_pipe*1000:8.1f} us  "
          f"(Saliency + {mean_fwd} x BNN/Frame, est.)")

    # ---- Verhaeltnisse ----
    base = rows[0][1]
    print(f"\n{'='*70}")
    print(f"  LAUFZEIT-VERHAELTNIS (Basis = {rows[0][0]})")
    print(f"{'='*70}")
    for name, r in rows:
        print(f"  {name:<32} x{r['ms']/base['ms']:6.2f}  "
              f"params={r['params']/base['params']:8.1f}  "
              f"MACs={r['macs']/base['macs']:8.1f}")

    # ---- Groessen ----
    print(f"\n{'='*70}")
    print(f"  MODELGROESSE geschaetzt (f32: 4B/Param)")
    print(f"{'='*70}")
    for name, r in rows:
        kb = r["params"] * 4 / 1024
        print(f"  {name:<32} {kb:10.1f} kB")

    # ---- Qualitaet: eigene Metriken ----
    print(f"\n{'='*70}")
    print(f"  QUALITAET (unsere Aufgabe, Val-Split)")
    print(f"{'='*70}")
    print("  Tile: AUROC / AP / recall@prec0.5")
    print("  End2End (gate=0.5): precision / recall / img_tp_rate / mean_fwd")
    q = {
        "MLP-Saliency":     (0.932, 0.799, 0.915, 0.284, 0.564, 0.600, 6.54),
        "Bottleneck-Sal":   (0.988, 0.957, 0.998, 0.326, 0.552, 0.626, 4.11),
    }
    for name, (auc, ap, rc, pr, re, itr, fwd) in q.items():
        print(f"  {name:<18} AUC={auc:.3f} AP={ap:.3f} recall={rc:.3f}"
              f" | prec={pr:.3f} rec={re:.3f} img_tp={itr:.3f} fwd={fwd:.2f}")
    print("\n  Referenz MobileNetV2: ImageNet-Classifier (top-1 ~86.4%),")
    print("  nicht faechergleich mit unserer 8x8-Tile-Saliency-Aufgabe.")
    print("  AI-Thinker/ESP-WHO: eigenes MTMN-FaceDetNet (s. Exporte),")
    print("  Laufzeit auf ESP32 separat messbar.")


if __name__ == "__main__":
    main()