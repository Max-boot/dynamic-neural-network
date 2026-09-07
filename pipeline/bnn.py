"""
Stufe 4: BNN (Bayesian Neural Network, MC-Dropout) fuer die Ziffern-Klassifikation.

Architektur: kleines Conv-MLP mit Dropout-Layern. Unsicherheit via
Monte-Carlo-Dropout bei der Inferenz: S Forward-Paesse mit aktivem Dropout ->
Mittelwert mu und Varianz sigma2 der softmax-Wahrscheinlichkeiten.

Klassen: 0-9 (Ziffern) + 10 (Hintergrund / kein Objekt).
Zweck: Region-Crops (28x28) der Stufe 3 bewerten; Ablehnung von Nicht-Ziffern
       sowohl durch die Hintergrund-Klasse als auch durch hohe sigma2.

Optionaler Bounding-Box-Head (Stufe 5): aus demselben Feature-Vektor wird die
digitale Box (cx, cy, w, h, normalisiert zum Crop) regressiert. Damit entsteht
eine multitask-Detektion: Klasse (0-9/10) + lokale Lokalisierung in einem
Forward-Pass - genau das, was fuer ESP32-CAM relevant ist.
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

BOX_DUMMY = 9.0  # Quadrat-Kantenlaenge implizit (nur semantisch in Doku)


class BNN(nn.Module):
    """Conv-MLP mit Dropout -> 11 Klassen; optional Box-Head; MC-Dropout."""

    def __init__(self, n_class=11, dropout=0.3, c1=24, c2=48, hid=96,
                 box_head=True):
        super().__init__()
        self.box_head_on = box_head
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=dropout)
        # nach 2x Pooling: 28 -> 7
        self.fc1 = nn.Linear(c2 * 7 * 7, hid)
        self.fc2 = nn.Linear(hid, n_class)
        if box_head:
            self.fc3 = nn.Linear(hid, 4)
            # Initialisierung: cx=cy=0.5 (sigmoid), w=h~0.25 (exp(ln 0.25))
            nn.init.normal_(self.fc3.weight, 0, 0.01)
            nn.init.constant_(self.fc3.bias[0], 0.0)
            nn.init.constant_(self.fc3.bias[1], 0.0)
            nn.init.constant_(self.fc3.bias[2], numpy.log(0.25))
            nn.init.constant_(self.fc3.bias[3], numpy.log(0.25))
        self.dropout_p = dropout

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)          # Dropout nur am Feature-Kopf (aktiv im Train)
        logits = self.fc2(x)
        if self.box_head_on:
            box = self.fc3(x)
            return logits, box
        return logits

    @torch.no_grad()
    def predict_mc(self, x, S=8, device="cuda"):
        """
        MC-Sampling: x [B,1,28,28] -> (mu [B,11], sigma2 [B,11]).
        Dropout bleibt waehrend Inferenz AN (self.train()).
        """
        self.to(device).train()
        x = x.to(device)
        probs = []
        for _ in range(S):
            logits = self(x)[0] if self.box_head_on else self(x)
            probs.append(F.softmax(logits, dim=1))
        p = torch.stack(probs, dim=0)    # [S,B,11]
        mu = p.mean(dim=0)
        sigma2 = p.var(dim=0)
        return mu.cpu(), sigma2.cpu()

    @torch.no_grad()
    def predict_mc_box(self, x, S=8, device="cuda"):
        """
        MC-Sampling mit Box-Head: x [B,1,28,28]
        -> mu (cls=11), sigma2 (cls=11), box_mu (B,4), box_sigma2 (B,4)
        box-Mittelwerte auf Sigmoid/Exp kodiert dekodiert: (cx,cy,w,h) in [0,1].
        """
        self.to(device).train()
        x = x.to(device)
        p_cls, p_boxc, p_boxe = [], [], []
        for _ in range(S):
            logits, boxr = self(x)
            p_cls.append(F.softmax(logits, dim=1))
            cx = torch.sigmoid(boxr[:, 0])
            cy = torch.sigmoid(boxr[:, 1])
            w = torch.exp(boxr[:, 2])
            h = torch.exp(boxr[:, 3])
            p_boxc.append(torch.stack([cx, cy], dim=1))
            p_boxe.append(torch.stack([w, h], dim=1))
        pc = torch.stack(p_cls, dim=0)
        pbc = torch.stack(p_boxc, dim=0)
        pbe = torch.stack(p_boxe, dim=0)
        mu = pc.mean(0)
        sigma2 = pc.var(0)
        # Box: Mittelwerte + Varianz
        mu_c = pbc.mean(0)
        var_c = pbc.var(0)
        mu_e = pbe.mean(0)
        var_e = pbe.var(0)
        box_mu = torch.cat([mu_c, mu_e], dim=1)          # (B,4): cx,cy,w,h
        box_var = torch.cat([var_c, var_e], dim=1)       # (B,4)
        return mu.cpu(), sigma2.cpu(), box_mu.cpu(), box_var.cpu()


def build_box_targets(y_cls):
    """
    Dummy: wird in bnn_data benoetigt, hier nur als Validierung-Helfer.
    Echte Targets kommen aus dem Datensatz.
    """
    return torch.zeros(y_cls.shape[0], 4)


def train_bnn(model, loader, val_loader, epochs=20, lr=1e-3, device="cuda",
              box_weight=3.0):
    """
    Multitask-Training: CE (Klassen, gewichtet gegen BG) + SmoothL1 (Box).
    Datensatz liefert (x, y_cls, y_box) pro Sample; fuer BG-Klassen ist die
    Box ignoriert (mask). Liefert (train_loss, val_acc, val_box_ok_fraction).
    """
    model.to(device)
    # Klassen-Gewichte aus Trainingsverteilung
    import collections
    counts = collections.Counter()
    for _, yb, _ in loader:
        for y in yb.tolist():
            counts[int(y)] += 1
    tot = sum(counts.values())
    w = numpy.zeros(model.fc2.out_features)
    for c, cnt in counts.items():
        w[c] = numpy.sqrt(tot / (len(counts) * cnt))
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)
    tr_losses, val_accs = [], []
    for ep in range(epochs):
        model.train()
        tot_l, nb = 0.0, 0
        for xb, yb, bb in loader:
            xb, yb, bb = (xb.to(device), yb.to(device), bb.to(device))
            opt.zero_grad()
            logits, box = model(xb)
            ce = F.cross_entropy(logits, yb, weight=weights)
            # Box-Loss nur bei Ziffern-Klassen
            mask = (yb < 10).unsqueeze(1)
            bx_pred = torch.cat([torch.sigmoid(box[:, :2]),
                                 torch.exp(box[:, 2:])], dim=1)
            sl = F.smooth_l1_loss(bx_pred * mask.float(), bb * mask.float(),
                                  reduction="sum") / max(1, mask.sum().item())
            loss = ce + box_weight * sl
            loss.backward()
            opt.step()
            tot_l += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        sched.step()
        tr_losses.append(tot_l / nb)
        # Val: Acc + Box-Trefferquote auf Test-Negativen mit "wenig Abweichung"
        model.eval()
        acc = 0.0
        nv = 0
        nBox = 0
        boxHit = 0
        with torch.no_grad():
            for xb, yb, bb in val_loader:
                xb, yb, bb = (xb.to(device), yb.to(device), bb.to(device))
                logits, box = model(xb)
                pred = logits.argmax(1)
                acc += (pred == yb).sum().item()
                nv += xb.shape[0]
                mask = yb < 10
                if mask.any():
                    m = mask.nonzero(as_tuple=False).view(-1)
                    bp = torch.cat([torch.sigmoid(box[m, :2]),
                                    torch.exp(box[m, 2:])], dim=1)
                    # Treffer: Zentrum innerhalb +-0.15, Groesse +-0.15
                    ok = (torch.abs(bp[:, 0] - bb[m, 0]) < 0.15) \
                         & (torch.abs(bp[:, 1] - bb[m, 1]) < 0.15) \
                         & (torch.abs(bp[:, 2] - bb[m, 2]) < 0.15) \
                         & (torch.abs(bp[:, 3] - bb[m, 3]) < 0.15)
                    boxHit += ok.sum().item()
                    nBox += mask.sum().item()
        v = acc / nv
        val_accs.append(v)
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:02d}/{epochs} loss={tr_losses[-1]:.4f} "
                  f"val_acc={v:.4f} box_hit={boxHit/max(1,nBox):.3f}")
    return tr_losses, val_accs