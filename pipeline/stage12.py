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


def to_model_input(imgs):
    """Scene-Bilder -> [N,C,128,128] float32. Akzeptiert 2D [N,128,128] oder
    Channel-nach-hinten [N,128,128,C] (RGB)."""
    a = numpy.asarray(imgs, dtype=numpy.float32)
    if a.ndim == 3:                      # [N,128,128] grayscale
        return a[:, None, :, :]
    if a.ndim == 4:                      # [N,128,128,C] channel-last
        if a.shape[-1] in (1, 3):
            return numpy.transpose(a, (0, 3, 1, 2))
    raise ValueError(f"unexpected images shape {a.shape}")


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


class TileStatsPerChannel(nn.Module):
    """Wie TileStats, aber die Kanaldimension bleibt erhalten: je 16x16-Kachel
    mean/max/var PRO Conv-Kanal -> [B, GRID*GRID, C*3]. Bei C=4 => 12 Features.

    Feature-Reihenfolge (fest, gespiegelt in nn.cpp + sim_pipeline.py):
        index = c*3 + s,  c in 0..C-1,  s in {0:mean, 1:max, 2:var}
        -> [c0_mean, c0_max, c0_var, c1_mean, ...]
    Instanz-Norm je der C*3 Feature-Spalten unabhaengig ueber die 64 Kacheln.
    """
    def __init__(self, tile=TILE, grid=GRID):
        super().__init__()
        self.tile = tile
        self.grid = grid

    def forward(self, feat):  # [B,C,128,128]
        B, C, H, W = feat.shape
        g = self.grid
        t = self.tile
        # [B,C,g,t,g,t] -> [B,g,g,C,t,t] -> [B,g,g,C,t*t] (Kanal bleibt getrennt)
        f = feat.view(B, C, g, t, g, t).permute(0, 2, 4, 1, 3, 5)
        f = f.reshape(B, g, g, C, t * t)
        mean = f.mean(dim=-1)                 # [B,g,g,C]
        mx = f.max(dim=-1).values             # [B,g,g,C]
        var = f.var(dim=-1, unbiased=False)   # [B,g,g,C]
        # stack -> [B,g,g,C,3] -> reshape [B,g,g,C*3] : Reihenfolge c*3+s
        stats = torch.stack([mean, mx, var], dim=-1)      # [B,g,g,C,3]
        F_ = C * 3
        stats = stats.reshape(B, g * g, F_)               # [B,64,C*3]
        # Instanz-Norm je Feature-Spalte ueber die Kacheln (ddof=1, clamp +-3)
        s = stats.permute(0, 2, 1)                         # [B,C*3,64]
        s = (s - s.mean(dim=2, keepdim=True)) / (s.std(dim=2, keepdim=True) + 1e-5)
        s = s.clamp(-3.0, 3.0)
        return s.permute(0, 2, 1)                          # [B,64,C*3]


class MLPHead(nn.Module):
    """Saliency-Kopf: 12 -> 16 -> 1 (ReLU-Hidden, roher Logit-Ausgang).

    Eingabe:  [B, 12]  (per-Kanal mean/max/var, instanz-normalisiert, clamp +-3)
    Ausgabe:  [B, 1]   (Saliency-Logit; aussen Sigmoid anwenden)
    Ersetzt den ANFIS: 0 Transzendente im Hidden (ReLU), deterministisch,
    exakt int-spiegelbar C<->Sim.
    """
    def __init__(self, n_in=12, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):                      # [B,12] -> [B,1]
        return self.fc2(F.relu(self.fc1(x)))


class ConvMLPSaliency(nn.Module):
    """Stage1+2 (MLP-Variante): [B,C,128,128] -> saliency [B,GRID*GRID] (Logits).

    Wie ConvANFISSaliency, aber per-Kanal-TileStats (12 Features) + MLPHead statt
    ANFIS. Dies ist das FUER DEN ESP32 deployte RGB-Modell (in_ch=3 im Training).
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.conv_stack = ConvStack(in_ch=in_ch, out_ch=4)
        self.tile_stats = TileStatsPerChannel()
        self.mlp = MLPHead(n_in=12, hidden=16)

    def forward(self, x):
        feat = self.conv_stack(x)              # [B,4,128,128]
        stats = self.tile_stats(feat)          # [B,64,12]
        B = stats.shape[0]
        flat = stats.reshape(-1, 12)           # [B*64,12]
        logits = self.mlp(flat)                # [B*64,1]
        return logits.reshape(B, GRID * GRID)  # [B,64]

    @torch.no_grad()
    def extract_features(self, x):
        """Liefert [B,64,12] (per-Kanal mean/max/var, instanz-normalisiert)."""
        feat = self.conv_stack(x)
        return self.tile_stats(feat)


# ---------------------------------------------------------------------------
# Linear Bottleneck (MobileNetV2) Saliency
# ---------------------------------------------------------------------------

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block.

    expand -> depthwise 3x3 -> project (LINEAR bottleneck, no activation).
    The key insight (Sandler et al. 2018): applying ReLU in the narrow
    projection collapses the low-dimensional manifold. The output of the
    project layer is therefore kept LINEAR. ReLU is applied only in the
    wide expanded space.
    """
    def __init__(self, in_ch, out_ch, expand, stride=1):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        # Expand: 1x1 conv -> BN -> ReLU  (wide space, safe for ReLU)
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        # Depthwise: 3x3 groups=hidden -> BN -> ReLU
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                             groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        # Project: 1x1 conv -> BN  (LINEAR bottleneck, no activation!)
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


class ConvBottleneckSaliency(nn.Module):
    """Saliency network with MobileNetV2-style linear bottleneck blocks.

    Replaces ConvStack + TileStats + MLP with a learned convolutional
    pipeline. Key difference: the narrow projection layers use NO activation
    (linear bottleneck), preserving information that ReLU would destroy.

    Input:  [B, in_ch, 128, 128]  (scene, RGB or grayscale)
    Output: [B, GRID*GRID]         (saliency logits, 8x8 grid)

    Approximate param count: ~1920 floats (7.7 KB) vs MLP's 741 (2.9 KB).
    Still compact enough for ESP32 PSRAM deployment with BN folding.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        # Stride-2 Stem: 128 -> 64 direkt, spart Full-Res-Convs deutlich.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        # Downsample: 64 -> 32 -> 16 -> 8
        self.ir1 = InvertedResidual(8, 8, expand=2, stride=2)    # [8, 32, 32]
        self.ir2 = InvertedResidual(8, 12, expand=2, stride=2)   # [12, 16, 16]
        self.ir3 = InvertedResidual(12, 12, expand=2, stride=2)  # [12, 8, 8]
        self.ir4 = InvertedResidual(12, 8, expand=2, stride=1)   # [8, 8, 8]
        # Linear head (no activation! the final linear bottleneck)
        self.head = nn.Conv2d(8, 1, 1, bias=False)               # [1, 8, 8]

    def forward(self, x):
        x = self.stem(x)
        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        return self.head(x).flatten(1)                           # [B, 64]


class ConvANFISSaliency(nn.Module):
    """Stage1+2 kombiniert: [B,C,128,128] -> saliency [B,GRID,GRID] (Logits)."""
    def __init__(self, n_in=3, n_mf=5, sigma_init=0.9, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.conv_stack = ConvStack(in_ch=in_ch, out_ch=4)
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
    X0 = to_model_input(train_imgs)                        # [N,C,128,128]
    X = torch.from_numpy(X0).to(device)
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
    X0 = to_model_input(imgs)
    X = torch.from_numpy(X0).to(device)
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