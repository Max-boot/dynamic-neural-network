"""
Stufe 1+2: Gemeinsames Modell Conv-Feature-Extraktor + ANFIS-Saliency.

Design (end-to-end differenzierbar):
  - Stage1: kleiner Conv-Stack (2x Conv3x3) auf 128x128 -> C-Kanal-Feature-Map.
  - Tile-Statistik: je 16x16-Kachel (8x8-Grid) mean/max/var ueber [16,16,C].
  - Instanz-Normalisierung der 3 Statistiken ueber die 64 Kacheln je Bild
    (robust gegen Frequenzdrift; Haelt ANFIS-Input im Zielfenster).
  - Stage2: ANFIS(3 Eingaenge, 5 MF = 125 Regeln) -> [B,64] Logit -> Sigmoide.
  - Verlust: BCEWithLogits (pos_weight fuer Klassenungleichgewicht).

FC: features von Stage1 werden an Stage3 (Decision Tree) weitergereicht;
der Tree sieht [saliency, mean, max, var].
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from anfis import ANFIS

TILE = 16
GRID = 8


class ConvStack(nn.Module):
    """
    Feature-Extraktor: 128x128 -> [B, C, 128, 128].
    Klein gehalten (Kanten/Helligkeitsstruktur-Primitiven), damit ANFIS die
    Kernlast traegt. C=4 reicht fuer mean/max/var-Aggregation aus.
    """
    def __init__(self, in_ch=1, out_ch=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [B,1,128,128]
        x = self.relu(self.conv1(x))  # [B,8,128,128]
        x = self.relu(self.conv2(x))  # [B,C,128,128]
        return x


class TileStats(nn.Module):
    """Berechnet je 16x16-Kachel mean/max/var ueber [16,16,C] -> [B,GRID*GRID,3].
    Differenzierbar; Instanz-Norm ueber Kacheln haelt Werte im stabilen Bereich.
    """
    def __init__(self, tile=TILE, grid=GRID):
        super().__init__()
        self.tile = tile
        self.grid = grid

    def forward(self, feat):  # [B,C,128,128]
        B, C, H, W = feat.shape
        g = self.grid
        t = self.tile
        # [B, C, g, t, g, t] -> [B, g, g, C*t*t]
        f = feat.view(B, C, g, t, g, t).permute(0, 2, 4, 1, 3, 5)
        f = f.reshape(B, g, g, C * t * t)
        mean = f.mean(dim=-1)          # [B,g,g]
        mx = f.max(dim=-1).values      # [B,g,g]
        var = f.var(dim=-1, unbiased=False)  # [B,g,g]
        stats = torch.stack([mean, mx, var], dim=-1)  # [B,g,g,3]
        stats = stats.view(B, g * g, 3)              # [B,64,3]
        # Instanz-Norm ueber Kachel-Dim (pro Kanal je Bild gemittelt)
        s = stats.permute(0, 2, 1)                   # [B,3,64]
        s = (s - s.mean(dim=2, keepdim=True)) / (s.std(dim=2, keepdim=True) + 1e-5)
        s = s.clamp(-3.0, 3.0)
        return s.permute(0, 2, 1)                    # [B,64,3]


class ConvANFISSaliency(nn.Module):
    """Stage1+2 kombiniert: [B,1,128,128] -> saliency [B,GRID,GRID] (Logits)."""
    def __init__(self, n_in=3, n_mf=5, sigma_init=0.9):
        super().__init__()
        self.conv_stack = ConvStack(in_ch=1, out_ch=4)
        self.tile_stats = TileStats()
        self.anfis = ANFIS(n_in=n_in, n_mf=n_mf,
                           center_range=(-3.0, 3.0), sigma_init=sigma_init)

    def forward(self, x):
        feat = self.conv_stack(x)              # [B,4,128,128]
        stats = self.tile_stats(feat)          # [B,64,3]
        B = stats.shape[0]
        flat = stats.reshape(-1, 3)            # [B*64, 3] fuer ANFIS
        logits = self.anfis(flat)              # [B*64, 1]
        return logits.reshape(B, GRID * GRID)  # [B,64]

    @torch.no_grad()
    def extract_features(self, x):
        """Liefert [B,64,3] (mean/max/var, instanz-normalisiert) fuer Stage3/Tree."""
        feat = self.conv_stack(x)
        return self.tile_stats(feat)


def pos_weight_from(tile_labels):
    """tile_labels: [N,64] 0/1 -> BCEWithLogits pos_weight."""
    tl = torch.as_tensor(tile_labels, dtype=torch.float32)
    p = tl.sum()
    n = tl.numel() - p
    return (n / p).clamp(max=50.0)


def train_stage12(model, train_imgs, train_tiles, epochs=40, batch=64,
                  lr=1e-3, device="cuda", verbose=True):
    model.to(device)
    pw = pos_weight_from(train_tiles).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(train_imgs).unsqueeze(1).to(device)   # [N,1,128,128]
    Y = torch.from_numpy(train_tiles).reshape(X.shape[0], -1).float().to(device)  # [N,64]
    N = X.shape[0]
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / N)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:02d}/{epochs}  loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def eval_stage12(model, imgs, tiles, device="cuda"):
    """Tile-Level-Metriken: AUROC, AP, Recall@Precision==0.5."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.to(device).eval()
    X = torch.from_numpy(imgs).unsqueeze(1).to(device)
    Y = tiles
    preds = []
    bs = 128
    for i in range(0, X.shape[0], bs):
        logits = model(X[i:i + bs]).cpu().numpy()
        preds.append(logits)
    p = numpy.concatenate(preds).ravel()
    y = Y.ravel()
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    # Recall bei der hoechsten Precision >= 0.5 (fuer Tree-Recall-Kalibrierung)
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    recall_at = 0.0
    # max. Recall, den wir bei Precision >= 0.5 erreichen (Tree-Recall-Ziel)
    good = numpy.where(pr >= 0.5)[0]
    if len(good):
        recall_at = rc[good].max()
    return {"auc": float(auc), "ap": float(ap),
            "recall@prec0.5": float(recall_at)}