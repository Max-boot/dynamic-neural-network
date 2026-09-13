"""
Training Gesichtspipeline (Stufe 1+2 Saliency + Stufe 4+5 BNN) auf WIDER-Szenen.

  - Stage12: ConvMLPSaliency (per-Kanal mean/max/var -> MLP 12-16-1) end-to-end
    BCE auf 8x8-Tiles.
  - BNN: n_class=2 (0=Gesicht, 1=Hintergrund), Box-Head (cx,cy,w,h).
    Multitask-Loss: CE(gewichtet) + SmoothL1(Box nur bei Gesicht).
Speichert conv_mlp_saliency_face.pt + bnn_mc_box_face.pt + Kurven/CSV.
"""
import errno
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_common import load_scene_split, boxes_to_tiles
from stage12 import ConvMLPSaliency, train_stage12, eval_stage12
from bnn import BNN
from face_data import build_face_datasets

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SCENES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "wider_scenes")

C1, C2, HID = 40, 80, 192
IN_CH = 3
EPOCHS_S12 = 40
EPOCHS_BNN = 70


def train_bnn_face(model, loader, val_loader, epochs=70, lr=1e-3,
                   device="cuda", box_weight=3.0):
    model.to(device)
    import collections
    counts = collections.Counter()
    for _, yb, _ in loader:
        for y in yb.tolist():
            counts[int(y)] += 1
    tot = sum(counts.values())
    n_cls = len(counts)
    w = numpy.zeros(model.fc2.out_features)
    for c, cnt in counts.items():
        w[c] = numpy.sqrt(tot / (n_cls * cnt))
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)
    tr_losses, val_accs, val_face_accs, val_box_hits = [], [], [], []
    for ep in range(epochs):
        model.train()
        tot_l, nb = 0.0, 0
        for xb, yb, bb in loader:
            xb, yb, bb = (xb.to(device), yb.to(device), bb.to(device))
            opt.zero_grad()
            logits, box = model(xb)
            ce = F.cross_entropy(logits, yb, weight=weights)
            mask = (yb == 0).unsqueeze(1)              # nur Gesicht
            bx_pred = torch.cat([torch.sigmoid(box[:, :2]),
                                 torch.exp(box[:, 2:])], dim=1)
            sl = F.smooth_l1_loss(bx_pred * mask.float(),
                                  bb * mask.float(),
                                  reduction="sum") / max(1, mask.sum().item())
            loss = ce + box_weight * sl
            loss.backward()
            opt.step()
            tot_l += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        sched.step()
        tr_losses.append(tot_l / nb)
        model.eval()
        acc = f_acc = boxHit = nBox = nv = nf = 0
        with torch.no_grad():
            for xb, yb, bb in val_loader:
                xb, yb, bb = (xb.to(device), yb.to(device), bb.to(device))
                logits, box = model(xb)
                pred = logits.argmax(1)
                acc += (pred == yb).sum().item()
                nv += xb.shape[0]
                face = yb == 0
                if face.any():
                    f_acc += (pred[face] == yb[face]).sum().item()
                    nf += face.sum().item()
                m = face
                if m.any():
                    mi = m.nonzero(as_tuple=False).view(-1)
                    bp = torch.cat([torch.sigmoid(box[mi, :2]),
                                    torch.exp(box[mi, 2:])], dim=1)
                    ok = (torch.abs(bp[:, 0] - bb[mi, 0]) < 0.15) \
                         & (torch.abs(bp[:, 1] - bb[mi, 1]) < 0.15) \
                         & (torch.abs(bp[:, 2] - bb[mi, 2]) < 0.15) \
                         & (torch.abs(bp[:, 3] - bb[mi, 3]) < 0.15)
                    boxHit += ok.sum().item()
                    nBox += m.sum().item()
        v = acc / max(1, nv)
        fv = f_acc / max(1, nf)
        bh = boxHit / max(1, nBox)
        val_accs.append(v); val_face_accs.append(fv); val_box_hits.append(bh)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:02d}/{epochs} loss={tr_losses[-1]:.4f} "
                  f"val_acc={v:.4f} face_acc={fv:.4f} box_hit={bh:.3f}")
    return tr_losses, val_accs, val_face_accs, val_box_hits


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # ---------- Stufe 1+2 ----------
    tr = load_scene_split("train", base=SCENES)
    te = load_scene_split("val", base=SCENES)
    tr_imgs, tr_tiles = tr["images"], boxes_to_tiles(tr["boxes"])
    te_imgs, te_tiles = te["images"], boxes_to_tiles(te["boxes"])
    print(f"Szenen Train: {tr_imgs.shape}  Val: {te_imgs.shape}  "
          f"pos-rate={tr_tiles.mean():.3f}")

    torch.manual_seed(42)
    saliency = ConvMLPSaliency(in_ch=IN_CH)
    losses = train_stage12(saliency, tr_imgs, tr_tiles, epochs=EPOCHS_S12,
                           batch=64, lr=1e-3, device=dev)
    te_met = eval_stage12(saliency, te_imgs, te_tiles, device=dev)
    print("Stage12 Val:", {k: round(v, 3) for k, v in te_met.items()})
    torch.save({"model": saliency.state_dict(), "metrics": te_met,
                "losses": losses},
               os.path.join(MODELS, "conv_mlp_saliency_face.pt"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, marker="o")
    ax.set(title="Stage12 Saliency (Gesichter) - BCE", xlabel="Epoche")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "saliency_face_loss.png"), dpi=150)
    plt.close(fig)

    # ---------- Stufe 4+5 ----------
    tr_ds, te_ds = build_face_datasets(seed=42, base=SCENES,
                                       max_pos_window=3, max_pos_tile=3,
                                       neg_per_img=6)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"BNN-Crops Train: {len(tr_ds)}  Val: {len(te_ds)}")

    torch.manual_seed(42)
    bnn = BNN(n_class=2, dropout=0.3, box_head=True, c1=C1, c2=C2, hid=HID,
              in_ch=IN_CH)
    print("BNN-Parameter:", sum(p.numel() for p in bnn.parameters()))
    b_losses, b_accs, b_faccs, b_bhits = train_bnn_face(
        bnn, tr_ld, te_ld, epochs=EPOCHS_BNN, lr=1e-3, device=dev,
        box_weight=3.0)
    torch.save({"model": bnn.state_dict(), "losses": b_losses,
                "val_accs": b_accs, "val_face_accs": b_faccs,
                "val_box_hits": b_bhits, "c1": C1, "c2": C2, "hid": HID},
               os.path.join(MODELS, "bnn_mc_box_face.pt"))
    print(f"Final val_acc={b_accs[-1]:.4f} face_acc={b_faccs[-1]:.4f} "
          f"box_hit={b_bhits[-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(b_losses)
    ax[0].set(title="BNN Face Training-Loss (CE+Box)", xlabel="Epoche")
    ax[0].grid(alpha=0.3)
    ax[1].plot(b_accs, label="val_acc (alle)")
    ax[1].plot(b_faccs, label="val_acc (Gesicht)")
    ax[1].set(title="BNN Val-Accuracy", xlabel="Epoche")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "bnn_face_training.png"), dpi=150)
    plt.close(fig)

    import csv
    with open(os.path.join(RESULTS, "face_train_results.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["s12_auc", f"{te_met['auc']:.4f}"])
        w.writerow(["s12_ap", f"{te_met['ap']:.4f}"])
        w.writerow(["bnn_final_val_acc", f"{b_accs[-1]:.4f}"])
        w.writerow(["bnn_final_face_acc", f"{b_faccs[-1]:.4f}"])
        w.writerow(["bnn_final_box_hit", f"{b_bhits[-1]:.3f}"])
        w.writerow(["bnn_params",
                    sum(p.numel() for p in bnn.parameters())])
    print("Fertig ->", RESULTS, "/", MODELS)


if __name__ == "__main__":
    main()