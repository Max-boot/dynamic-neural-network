"""
ANFIS-Modul (Adaptive Neuro-Fuzzy Inference System), Sugeno Typ-3, differenzierbar.

Eingabe:   [B, n_in]  (n_in = 3 Standard-Inputs: mean, max, var je Kachel)
Ausgabe:   [B, 1]     (singulaerer Saliency-Logit; aussen Sigmoid anwenden)

Architektur (Sugeno type-3 / first-order consequent):
  Schicht 1 Fuzzifizierung : je Input Gauß-MF  mu_ij = exp(-((x_i-c_ij)^2)/(2*s_ij^2))
  Schicht 2 Regeln         : w_r = prod_i mu_{i, rule}_i   (alle Kombinationen)
  Schicht 3 Normalisierung : wbar_r = w_r / (sum_r w_r + eps)
  Schicht 4 Konsequenten   : f_r = sum_k p_{r,k} * x_k   (x_0 = 1 Bias)
  Schicht 5 Ausgabe        : y = sum_r wbar_r * f_r

Trainierbare Parameter: MF-Zentren c, MF-Breiten s (softplus-parametrisiert),
                         Konsequenten-Matrix P [n_rules, n_in+1].

Regelanordnung wird einmalig per init aufgebaut (itertools.product), danach
vektorisiert ausgefuehrt -> GPU-freundlich.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANFIS(nn.Module):
    def __init__(self, n_in: int = 3, n_mf: int = 5,
                 center_range=(0.0, 2.0), sigma_init: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.n_in = n_in
        self.n_mf = n_mf
        self.eps = eps

        # Membership-Funktions-Zentren pro Input, gleichmaessig ueber den (erwarteten)
        # Feature-Bereich verteilt. sigma_init als Startbreite.
        c = torch.linspace(center_range[0], center_range[1], n_mf)
        # MFs unabhängig je Input initialisiert (gleicher Grid fuer alle Inputs ok)
        self.c = nn.Parameter(c.expand(n_in, n_mf).contiguous())
        # Breiten: learnable via softplus -> immer positiv; start bei sigma_init
        self.log_sigma = nn.Parameter(
            torch.full((n_in, n_mf), float(torch.log(torch.tensor(sigma_init)))))

        # Regel-Indizes: [n_rules, n_in] mit n_rules = n_mf^n_in
        rules = list(itertools.product(range(n_mf), repeat=n_in))
        self.register_buffer("rule_idx", torch.tensor(rules, dtype=torch.long))
        self.n_rules = len(rules)

        # Konsequenten-Parameter P [n_rules, n_in+1]
        p = torch.randn(self.n_rules, n_in + 1) * 0.5
        self.P = nn.Parameter(p)

    @property
    def sigma(self):
        return F.softplus(self.log_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_in] -> y: [B, 1] (Logit)."""
        B = x.shape[0]
        # (1) Mitgliedschaften: [B, n_in, n_mf]
        diff = x.unsqueeze(-1) - self.c.unsqueeze(0)      # [B, n_in, n_mf]
        sig = self.sigma.unsqueeze(0)                     # [B, n_in, n_mf]
        mu = torch.exp(-(diff ** 2) / (2 * sig ** 2 + self.eps))  # [B, n_in, n_mf]

        # (2) Feuer-Studien aller Regeln: [B, n_rules]
        #     mu[:, i, rule_idx[r, i]] -> gather
        ridx = self.rule_idx                            # [n_rules, n_in]
        # mu umformen: [B, n_in, n_mf] -> [n_in, B, n_mf] vereinfacht nicht;
        # nutze Broadcasting ueber index_select je Input.
        mu_i = [mu[:, i, :] for i in range(self.n_in)]  # je [B, n_mf]
        w = torch.ones(B, self.n_rules, device=x.device)
        for i in range(self.n_in):
            col = mu_i[i]                                # [B, n_mf]
            idx = ridx[:, i]                            # [n_rules]
            w = w * col[:, idx]                          # [B, n_rules]

        # (3) Normalisierung
        wbar = w / (w.sum(dim=1, keepdim=True) + self.eps)

        # (4) Konsequenten f_r = p0 + sum_k p_k x_k ; k=0 ist Bias x_0=1
        #     z = [1, x]  [B, n_in+1]
        ones = torch.ones(B, 1, device=x.device)
        z = torch.cat([ones, x], dim=1)                  # [B, n_in+1]
        f = z @ self.P.t()                               # [B, n_rules]

        # (5) Ausgabe
        y = (wbar * f).sum(dim=1, keepdim=True)          # [B, 1]
        return y

    def extra_repr(self):
        return (f"n_in={self.n_in}, n_mf={self.n_mf}, n_rules={self.n_rules}, "
                f"params_c={self.c.numel()}, params_P={self.P.numel()}")