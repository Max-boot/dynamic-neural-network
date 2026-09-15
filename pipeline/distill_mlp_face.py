"""
Stufe 4-Substitution: Distillation BNN(MD-Box)-Teacher -> kleiner Conv-MLP-Student.

Idee (User-Entscheid):
  - Studierender ist ein REINER Klassifikator (nur Gesicht/Hintergrund),
    KEIN Box-Head. Die Bounding-Box wird direkt aus dem adaptiven Fenster der
    Stufe 3 uebernommen ("Box = Fenster") - dadurch entfaellt der teure
    Box-Head (fc3) komplett und das Netz wird deutlich kleiner.
  - Der Teacher ist das trainierte BNN `bnn_mc_box_face.pt` (MC-Dropout).
    Distillation mit Soft-Targets (MC-Mittelwert der Softmax-Wahrscheinlich-
    keiten ueber S Forward-Paesse mit aktivem Dropout) + harte Labels gleich-
    zeitig: loss = soft_CE(Student, Teacher-MC-Mean, T) + w * CE(hart).

Architektur-Student: gleiche BNN-Klasse (Conv1->BN->Pool->Conv2->BN->Pool->
FC1->FC2) aber:
  - box_head=False            (kein Box-Head -> 4 Neuronen + fc3 weg)
  - kleinere c1/c2/hid        (Stellschrauben, siehe unten)
Dadurch bleiben die ESP32-Primitiven (conv3x3_same_relu, maxpool2, fc)
identisch zur bestehenden Firmware - es aendert sich nur die Blob-Groesse
und der Wegfall von fc3 (Global-Indexierung im Export/Verify/Firmware).

Speichert: models/student_mlp_face.pt (state_dict + Teacher-Infos + Metriken)
           results/distill_mlp_face_training.png / .csv

Kommandozeile:
  python distill_mlp_face.py [--epochs 40] [--lr 1e-3] [--temp 3.0]
                             [--c1 24] [--c2 40] [--hid 96] [--mc 8]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from bnn import BNN
from face_data import build_face_datasets, BG_CLASS

MODELS = os.path.join(_HERE, "models")
RESULTS = os.path.join(_HERE, "results")
TEACHER_CKPT = "bnn_mc_box_face.pt"
STUDENT_SAVE = "student_mlp_face.pt"

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def teacher_soft_probs(teacher, xb, S=8, device="cuda"):
    """MC-Mean der Softmax-Wahrscheinlichkeiten des Teachers.

    xb: [B,3,28,28] -> [B,2] float32 (nur Klassenkopf; Box wird ignoriert).
    BNN.predict_mc_box liefert (mu_cls, sigma2_cls, box_mu, box_var) - wir
    nehmen nur mu_cls -> das ist der "weiche" Distillations-Lehrer.
    """
    teacher.to(device).eval()
    xb = xb.to(device)
    mu, _sigma2, _bm, _bv = teacher.predict_mc_box(xb, S=S, device=device)
    return mu.float()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=3.0,
                    help="Temperatur fuer Soft-CE (hoeher = weichere Targets)")
    ap.add_argument("--c1", type=int, default=24)
    ap.add_argument("--c2", type=int, default=40)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--mc", type=int, default=8, help="MC-Samples Teacher")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # ---------------- Teacher ----------------
    teacher = BNN(n_class=2, box_head=False, c1=40, c2=80, hid=192, in_ch=3)
    ck = os.path.join(MODELS, TEACHER_CKPT)
    if not os.path.isfile(ck):
        raise SystemExit(
            f"Teacher-Checkpoint fehlt: {ck}\nBitte zuerst train_face.py "
            f"(Stufe 4+5) laufen lassen, bevor distills wird.")
    # Box-Head des Teachers einschalten (so wurde trainiert), dann laden.
    teacher = BNN(n_class=2, box_head=True, c1=40, c2=80, hid=192, in_ch=3)
    teacher.load_state_dict(torch.load(ck, map_location="cpu",
                                       weights_only=False)["model"])
    teacher.eval()
    print(f"Teacher geladen: {TEACHER_CKPT} ({sum(p.numel()
          for p in teacher.parameters()):,} Params)")

    # ---------------- Daten ----------------
    tr_ds, te_ds = build_face_datasets(seed=args.seed, max_pos_window=2,
                                       max_pos_tile=2, neg_per_img=3)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"Train-Crops: {len(tr_ds)}  Val-Crops: {len(te_ds)}")

    # ---------------- Student (MLP, kein Box-Head) ----------------
    student = BNN(n_class=2, box_head=False, c1=args.c1, c2=args.c2,
                  hid=args.hid, in_ch=3)
    stud_params = sum(p.numel() for p in student.parameters())
    print(f"Student-Architektur: c1={args.c1} c2={args.c2} hid={args.hid} "
          f"(box_head=False) -> {stud_params:,} Params")

    # Klassen-Gewichte (Gesicht selten) fuer die harte CE.
    import collections
    cnt = collections.Counter()
    for _, yb, _ in tr_ld:
        cnt.update(int(y) for y in yb.tolist())
    tot = sum(cnt.values())
    w = numpy.zeros(2, dtype=numpy.float32)
    for c, n in cnt.items():
        w[c] = numpy.sqrt(tot / (2 * n))
    weights = torch.tensor(w, dtype=torch.float32, device=DEV)
    print(f"  Klassen-Verteilung: {dict(cnt)}  CE-Weights: {w.tolist()}")

    student.to(DEV)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)

    tr_losses, val_accs, val_face_accs = [], [], []
    T = args.temp
    for ep in range(args.epochs):
        student.train()
        tot_l, nb = 0.0, 0
        for xb, yb, _bb in tr_ld:
            xb, yb = xb.to(DEV), yb.to(DEV)
            # Soft-Targets: MC-Mean des Teachers (Klasse + Box getrennt,
            # nur Klassenwahrscheinlichkeiten mit Temperatur-Teilung).
            soft = teacher_soft_probs(teacher, xb, S=args.mc, device=DEV)
            soft = F.softmax(torch.log(soft + 1e-9) / T, dim=1).to(DEV)
            logits = student(xb)
            sc = F.cross_entropy(
                logits / T, soft.argmax(1)) + \
                F.kl_div(F.log_softmax(logits / T, dim=1), soft,
                         reduction="batchmean") * (T * T)
            hc = F.cross_entropy(logits, yb, weight=weights)
            loss = sc + hc
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_l += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        sched.step()
        tr_losses.append(tot_l / max(1, nb))

        # ---------------- Val ----------------
        student.eval()
        acc = f_acc = nv = nf = 0
        with torch.no_grad():
            for xb, yb, _bb in te_ld:
                xb, yb = xb.to(DEV), yb.to(DEV)
                logits = student(xb)
                pred = logits.argmax(1)
                acc += (pred == yb).sum().item()
                nv += xb.shape[0]
                face = yb == 0
                if face.any():
                    f_acc += (pred[face] == yb[face]).sum().item()
                    nf += face.sum().item()
        va = acc / max(1, nv)
        vf = f_acc / max(1, nf)
        val_accs.append(va)
        val_face_accs.append(vf)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:02d}/{args.epochs} loss={tr_losses[-1]:.4f} "
                  f"val_acc={va:.4f} face_acc={vf:.4f}")

    # ---------------- Speichern + Report ----------------
    final_acc = val_accs[-1]
    final_face = val_face_accs[-1]
    torch.save({"model": student.state_dict(),
                "arch": {"c1": args.c1, "c2": args.c2, "hid": args.hid,
                         "box_head": False, "in_ch": 3},
                "teacher": TEACHER_CKPT, "temp": T, "mc": args.mc,
                "val_acc": final_acc, "val_face_acc": final_face,
                "params": stud_params,
                "losses": tr_losses, "val_accs": val_accs,
                "val_face_accs": val_face_accs},
               os.path.join(MODELS, STUDENT_SAVE))
    print(f"\nGespeichert: {STUDENT_SAVE}  val_acc={final_acc:.4f} "
          f"face_acc={final_face:.4f} params={stud_params:,}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(tr_losses)
    ax[0].set(title="Distill-Loss (Student)", xlabel="Epoche"); ax[0].grid(alpha=0.3)
    ax[1].plot(val_accs, label="val_acc")
    ax[1].plot(val_face_accs, label="val_face_acc")
    ax[1].set(title="Student Val-Accuracy", xlabel="Epoche")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "distill_mlp_face_training.png"), dpi=150)
    plt.close(fig)

    import csv
    with open(os.path.join(RESULTS, "distill_mlp_face_results.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["student_params", stud_params])
        w.writerow(["val_acc", f"{final_acc:.4f}"])
        w.writerow(["val_face_acc", f"{final_face:.4f}"])
        w.writerow(["c1", args.c1]); w.writerow(["c2", args.c2])
        w.writerow(["hid", args.hid]); w.writerow(["temp", T])
    print(f"Ergebnisse -> {os.path.join(RESULTS, 'distill_mlp_face_results.csv')}")


if __name__ == "__main__":
    main()
