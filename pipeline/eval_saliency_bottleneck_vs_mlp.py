"""
End-to-End-Vergleich: MLP-Saliency vs Linear-Bottleneck-Saliency.

Beide Saliency-Varianten durch dieselbe Detektions-Pipeline (Regionen ->
hybrid2-Fenster -> BNN 2 Klassen). Der BNN bleibt identisch; nur die
Saliency-Karte (und damit die Regionen/Fenster) unterscheidet sich.

Metriken je Variante: Precision, Recall (IoU>=0.5), img_tp_rate, mean_fwd.
"""
import os
import sys

import numpy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split
from stage12 import ConvMLPSaliency, ConvBottleneckSaliency, to_model_input
from bnn import BNN
from eval_face_pipeline import evaluate_pipeline_face

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SCENES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "wider_scenes")

IN_CH = 3
BNN_C1, BNN_C2, BNN_HID = 40, 80, 192


def saliency_map(model, imgs, dev):
    model.to(dev).eval()
    with torch.no_grad():
        X = torch.from_numpy(to_model_input(imgs)).to(dev)
        out = []
        for i in range(0, X.shape[0], 64):
            out.append(torch.sigmoid(model(X[i:i + 64]))
                       .cpu().numpy().reshape(-1, 8, 8))
        return numpy.concatenate(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    bnn = BNN(n_class=2, box_head=True, c1=BNN_C1, c2=BNN_C2, hid=BNN_HID,
              in_ch=IN_CH)
    bnn.load_state_dict(torch.load(
        os.path.join(MODELS, "bnn_mc_box_face.pt"),
        map_location="cpu", weights_only=False)["model"])

    te = load_scene_split("val", base=SCENES)
    imgs, gtb, gtl = te["images"], te["boxes"], te["labels"]
    print(f"Val-Szenen: {imgs.shape}")

    variants = {}

    sal = ConvMLPSaliency(in_ch=IN_CH)
    sal.load_state_dict(torch.load(
        os.path.join(MODELS, "conv_mlp_saliency_face.pt"),
        map_location="cpu", weights_only=False)["model"])
    variants["mlp"] = saliency_map(sal, imgs, dev)

    sal2 = ConvBottleneckSaliency(in_ch=IN_CH)
    sal2.load_state_dict(torch.load(
        os.path.join(MODELS, "saliency_bottleneck_exp.pt"),
        map_location="cpu", weights_only=False)["model"])
    variants["bottleneck"] = saliency_map(sal2, imgs, dev)

    import collections
    summary = collections.defaultdict(dict)
    for name, sal_all in variants.items():
        print(f"\n=== Saliency: {name} ===")
        for gate in (0.4, 0.5, 0.6, 0.7):
            m = evaluate_pipeline_face(bnn, sal_all, imgs, gtb, gtl,
                                       device=dev, gate=gate)
            summary[name][gate] = m
            print(f"--- gate={gate} ---")
            for k, v in m.items():
                print(f"  {k}: {round(v, 4)}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n{'='*70}")
    print("  VERGLEICH (MLP vs Bottleneck)")
    print(f"{'='*70}")
    for gate in (0.4, 0.5, 0.6, 0.7):
        a, b = summary["mlp"][gate], summary["bottleneck"][gate]
        d = "BOTTLENECK gewinnt" if b["recall"] > a["recall"] else "MLP gewinnt"
        print(f"  gate={gate}: precision {a['precision']:.4f} -> {b['precision']:.4f}, "
              f"recall {a['recall']:.4f} -> {b['recall']:.4f}  ({d})")

    print("\nFertig.")


if __name__ == "__main__":
    main()