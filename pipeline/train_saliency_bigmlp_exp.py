"""Fairer Vergleich: parametergleiches MLP (~3k) vs ConvBottleneckSaliency.

Der Linear-Bottleneck-Sprung (741->2992 Params) bringt +0.056 AUC. Um
Kapazitaet von Architektur zu trennen, wird die MLP-Variante auf ~3000
Parameter skaliert und neu trainiert (gleicher Seed/Setup).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, boxes_to_tiles
from stage12 import (ConvStack, TileStatsPerChannel, MLPHead,
                     train_stage12, eval_stage12)

SCENES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "wider_scenes")


class BigMLPSaliency(torch.nn.Module):
    """ConvStack(3->8->12) + TileStatsPerChannel(36 Feat.) + MLP(36->48->1)."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.conv_stack = ConvStack(in_ch=in_ch, out_ch=12)
        self.tile_stats = TileStatsPerChannel()
        self.mlp = MLPHead(n_in=36, hidden=48)

    def forward(self, x):
        feat = self.conv_stack(x)              # [B,12,128,128]
        stats = self.tile_stats(feat)          # [B,64,36]
        B = stats.shape[0]
        flat = stats.reshape(-1, 36)
        logits = self.mlp(flat)
        return logits.reshape(B, 64)


def main():
    tr = load_scene_split("train", base=SCENES)
    te = load_scene_split("val", base=SCENES)
    tr_imgs, tr_tiles = tr["images"], boxes_to_tiles(tr["boxes"])
    te_imgs, te_tiles = te["images"], boxes_to_tiles(te["boxes"])

    torch.manual_seed(42)
    model = BigMLPSaliency(in_ch=3)
    n = sum(p.numel() for p in model.parameters())
    print(f"Big-MLP params: {n} ({n*4} B f32)")
    t0 = time.time()
    losses = train_stage12(model, tr_imgs, tr_tiles, epochs=40,
                           batch=64, lr=1e-3, device="cuda")
    print(f"train sec: {time.time()-t0:.1f}")
    met = eval_stage12(model, te_imgs, te_tiles, device="cuda")
    print("met:", met)
    torch.save({"model": model.state_dict(), "losses": losses,
                "n_params": n, "metrics": met},
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "saliency_bigmlp_exp.pt"))


if __name__ == "__main__":
    main()