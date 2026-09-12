"""
RL-Pruner: das ML-LERNT selbst, wie es prunen soll (Policy-Gradient / REINFORCE)
==============================================================================
Bisherige Skripte nutzten einen FEST verdrahteten Meta-Scorer + feste Regel:
    prio = (1 - score) * Ersparnis
Das ist zwar ML beim Bewerten, aber KEIN ML bei der Entscheidung (welches
Neuron wird geprunt) -> keine echte Lern-Schleife.

Hier bauen wir einen WIRKLICH lernenden Pruner:
  - Eine Policy (kleines MLP) bekommt zu jedem aktiven Neuron 6 Features und
    gibt einen "Prune-Logit" aus.
  - Beim Pruning wird GREEDY geprunt (wichtigste Logits zuerst), aber mit
    gumbel-Stoerung, damit stochastische Aktionen entstehen (REINFORCE-Kontext).
  - Nach jedem vollstaendigen Prune-Durchlauf bis zum Synapsen-Budget wird das
    Netz kompaktiert + feingetunt und die Test-Accuracy gemessen.
  - REWARD = Test-Accuracy (bzw. Acc - Baseline). Per Policy-Gradient wird die
    Policy so trainiert, dass sie Aktionen mit hohem Reward wahrscheinlicher
    macht. Die Policy LERNT also aus Erfahrung, wie man prunen muss, um die
    Accuracy zu erhalten.

Der Trainingszweig speichert:
  - models/pruner_policy.pt            (trainierte Policy)
  - RESULT_rl_pruner_training.csv      (Episode-Verlauf: acc, syn, reward, loss)
  - RESULT_bild_rl_pruner_training.png (Accverlauf + Policy-Loss)

Der Auswertungszweig (--eval) nutzt die trainierte Policy deterministisch
(argmax-Logit) und vergleicht mit der festen (1-score)*s-Regel aus dem
budget_prune-Skript sowie mit einem Random-Pruner.
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
MT_PATH = r"D:\Dynamic Neural Networt (DNN)\Code\structural_ml_metatrain.py"

def _load_mt():
    spec = importlib.util.spec_from_file_location("mt", MT_PATH)
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt

mt = _load_mt()
device = mt.device
train_loader = mt.train_loader
test_loader = mt.test_loader
print(f"Device: {device}", flush=True)

LAYOUT = [256, 128, 64]
SYN_TARGET = 40000          # Budget (~ -83.5% von 242304)
MAX_STEPS_PER_EPISODE = 4000
WARM = 3                    # Epochen vor dem Pruning
FINETUNE = 3                # Epochen nach dem Kompaktieren
TOPK = 5                    # gumbel-Stichproben-Pool pro Schnitt

EPISODE_SEEDS = [42, 2024, 7, 100]   # verschiedene Netze je Episode
RL_EPOCHS = 10                       # Gradient-Updates je Episode
RL_LR = 1e-3


# ═════════════════════════════════════════════════════════════════════════
# Policy-Netz: Neuron-Features -> "Prune-Logit" (je hoeher, desto eher prunen)
# ═════════════════════════════════════════════════════════════════════════
class PrunerPolicy(nn.Module):
    def __init__(self, n_feat=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def collect_feats(model, scorer=None):
    """6 neuron-Features je aktives Neuron (wie fuer den Meta-Scorer)."""
    model.reset_activity_buffer()
    model.reset_class_stats()
    for xb, yb in list(train_loader)[:3]:
        xb, yb = xb.to(device), yb.to(device)
        model(xb)
        model.update_class_stats(xb, yb)
    corr = model.compute_correlation_matrix()
    return mt.extract_features(model, corr)          # (tot_hidden, 6)


def syn_savings_layer(model, L):
    """Ersparnis beim Entfernen EINES Neurons der Schicht L (aktuelle Groesse)."""
    counts = model.neuron_counts
    in_dim = 28 * 28 if L == 0 else counts[L - 1]
    out_dim = model.n_classes if L == model.n_layers - 1 else counts[L + 1]
    return in_dim + out_dim


def train_epochs(model, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def compact_dynamicmlp(src, keep_counts, keep_idx_lists):
    nc = mt.DynamicMLP(keep_counts).to(device)
    src_l = src.layers
    dst_l = nc.layers
    with torch.no_grad():
        for i in range(len(keep_counts)):
            ki = keep_idx_lists[i].to(device)
            if i == 0:
                dst_l[0].weight.copy_(src_l[0].weight[ki])
                dst_l[0].bias.copy_(src_l[0].bias[ki])
            else:
                k_prev = keep_idx_lists[i - 1].to(device)
                dst_l[i].weight.copy_(src_l[i].weight[ki][:, k_prev])
                dst_l[i].bias.copy_(src_l[i].bias[ki])
        k_last = keep_idx_lists[-1].to(device)
        dst_l[len(keep_counts)].weight.copy_(
            src_l[len(keep_counts)].weight[:, k_last])
        dst_l[len(keep_counts)].bias.copy_(src_l[len(keep_counts)].bias)
    return nc


def rl_prune(model, policy, feats, syn_target, temperature=0.05):
    """
    Stochastisches Greedy-Pruning nach den Policy-Logits:
      - In jedem Schritt: alle aktiven Neuronen, berechne Policy-Logit.
      - Nimm die TOPK Kandidaten nach Logit, ziehe einen gumbel-gewichteten
        (softmax/temperature) -> stochastische Aktion.
      - Prune das Neuron, summiere log p(a) fuer REINFORCE (mit Grad).
      - Bis active_synapses <= syn_target (oder kein Kandidat mehr).
    Rueckgabe: (keep_counts, sum_log_probs_tensor, steps)
    """
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(0.15 * o), 8) for o in orig]
    starts = model.hidden_start_indices
    sum_logp = torch.zeros((), device=device)   # acccumuliert mit Gradient
    steps = 0

    while model.active_synapses > syn_target and steps < MAX_STEPS_PER_EPISODE:
        counts = model.neuron_counts
        if all(c <= f for c, f in zip(counts, floor)):
            break
        L_idx = []
        feat_idx = []
        feats_batch = []
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue
            mask = model.neuron_masks[L].cpu().numpy()
            for nl in np.where(mask > 0)[0]:
                L_idx.append(L)
                feat_idx.append(int(nl))
                feats_batch.append(feats[starts[L] + nl])
        if not L_idx:
            break
        feats_batch = torch.from_numpy(np.array(feats_batch)).float().to(device)
        logits = policy(feats_batch)               # Torch-Tensor MIT Grad
        k = min(TOPK, len(logits))
        top_idx = torch.topk(logits, k).indices
        logp = F.log_softmax(logits[top_idx] / temperature, dim=0)
        noise = -torch.log(-torch.log(torch.rand(k, device=device) + 1e-8) + 1e-8)
        pick = int((logp.detach() + noise).argmax().item())
        t_i = int(top_idx[pick])
        sum_logp = sum_logp + logp[pick]
        gL, gnl = L_idx[t_i], feat_idx[t_i]
        mask = model.neuron_masks[gL].clone()
        mask[gnl] = 0.0
        setattr(model, f"neuron_mask_{gL}", mask)
        steps += 1
    return model.neuron_counts, sum_logp, steps


def run_episode(seed, policy):
    """Trainiert ein Netz (seed), prunt per Policy bis Budget, misst Acc.
    Rueckgabe: dict mit Acc, Synapsen, sum_logp (fuer REINFORCE)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = mt.DynamicMLP(LAYOUT).to(device)
    train_epochs(model, WARM)
    acc_before = mt.measure_accuracy(model)
    syn_before = model.active_synapses

    feats = collect_feats(model)
    keep_counts, sum_logp, steps = rl_prune(model, policy, feats, SYN_TARGET)
    syn_after_mask = model.active_synapses

    keep_idx_lists = [torch.nonzero(m > 0).squeeze(1).cpu()
                      for m in model.neuron_masks]
    cmod = compact_dynamicmlp(model, keep_counts, keep_idx_lists)
    syn_compact = cmod.active_synapses
    train_epochs(cmod, FINETUNE)
    acc_final = mt.measure_accuracy(cmod)

    return {
        "seed": seed,
        "acc_before": acc_before,
        "acc_final": acc_final,
        "syn_before": syn_before,
        "syn_compact": syn_compact,
        "syn_after_mask": syn_after_mask,
        "layers": keep_counts,
        "logp": sum_logp,
    }


# ═════════════════════════════════════════════════════════════════════════
# REINFORCE-Training
# ═════════════════════════════════════════════════════════════════════════
def train_rl():
    policy = PrunerPolicy(n_feat=6).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=RL_LR)
    history = []

    for ep in range(RL_EPOCHS):
        ep_logp = []
        ep_reward = []
        ep_acc = []
        ep_syn = []
        for seed in EPISODE_SEEDS:
            policy.eval()
            res = run_episode(seed, policy)
            reward = (res["acc_final"] - res["acc_before"]) / 100.0  # Acc-Delta
            ep_logp.append(res["logp"])
            ep_reward.append(reward)
            ep_acc.append(res["acc_final"])
            ep_syn.append(res["syn_compact"])
            print(f"  [Ep {ep+1}/{RL_EPOCHS}] seed={seed} "
                  f"acc={res['acc_before']:.2f}->{res['acc_final']:.2f} "
                  f"syn={res['syn_compact']:,} layers={res['layers']} "
                  f"R={reward:+.2f}", flush=True)

        # Policy-Update: minus (R - baseline) * sum(log p)
        total_logp = torch.stack(ep_logp).sum()
        loss = -(total_logp * np.mean(ep_reward))
        opt.zero_grad()
        loss.backward()
        opt.grad_clip = None
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()

        history.append({
            "episode": ep + 1,
            "acc_mean": float(np.mean(ep_acc)),
            "acc_best": float(np.max(ep_acc)),
            "syn_mean": float(np.mean(ep_syn)),
            "reward_mean": float(np.mean(ep_reward)),
            "loss": float(loss.item()),
        })
        print(f"  >>> Ep {ep+1}: acc_mittel={np.mean(ep_acc):.2f} "
              f"R_mittel={np.mean(ep_reward):+.2f} loss={loss.item():.3f}",
              flush=True)

    os.makedirs(os.path.join(_HERE, "models"), exist_ok=True)
    policy_path = os.path.join(_HERE, "models", "pruner_policy.pt")
    torch.save(policy.state_dict(), policy_path)
    print(f"\n  Policy gespeichert: {policy_path}", flush=True)
    _save_plot_history(history)
    _save_csv_history(history)
    return policy, history


def _save_plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = [h["episode"] for h in history]
    ax[0].plot(ep, [h["acc_mean"] for h in history], marker="o",
               label="Acc nach Prune")
    ax[0].plot(ep, [h["reward_mean"] for h in history], marker="s",
               label="Reward (Acc-Delta)", color="#1f77b4")
    ax[0].axhline(0.0, color="gray", ls="--")
    ax[0].set_xlabel("RL-Episode"); ax[0].set_ylabel("Test-Accuracy (%)")
    ax[0].set_title("RL-Pruner: Acc-Verhalten beim Lernen"); ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [h["loss"] for h in history], marker="o", color="#d62728")
    ax[1].set_xlabel("RL-Episode"); ax[1].set_ylabel("Policy-Loss")
    ax[1].set_title("REINFORCE Loss (minus Reward x log p)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(_HERE, "RESULT_bild_rl_pruner_training.png")
    fig.savefig(png, dpi=150)
    print(f"  Plot gespeichert: {png}", flush=True)


def _save_csv_history(history):
    csvp = os.path.join(_HERE, "RESULT_rl_pruner_training.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        for h in history:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in h.items()})
    print(f"  CSV gespeichert: {csvp}", flush=True)


# ═════════════════════════════════════════════════════════════════════════
# Auswertung: gelernte Policy (deterministisch) vs. feste Regel vs. Random
# ═════════════════════════════════════════════════════════════════════════
def eval_deterministic(model, policy, feats, syn_target):
    """Greedy-Pruning nach ARGMAX des Policy-Logits (keine Stoerung)."""
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(0.15 * o), 8) for o in orig]
    starts = model.hidden_start_indices
    steps = 0
    while model.active_synapses > syn_target and steps < MAX_STEPS_PER_EPISODE:
        counts = model.neuron_counts
        if all(c <= f for c, f in zip(counts, floor)):
            break
        best = None
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue
            mask = model.neuron_masks[L].cpu().numpy()
            for nl in np.where(mask > 0)[0]:
                f = torch.from_numpy(feats[starts[L] + nl]).float().to(device)
                logit = policy(f.unsqueeze(0)).item()
                if best is None or logit > best[0]:
                    best = (logit, L, int(nl))
        if best is None:
            break
        _, L, nl = best
        mask = model.neuron_masks[L].clone()
        mask[nl] = 0.0
        setattr(model, f"neuron_mask_{L}", mask)
        steps += 1
    return model.neuron_counts


def eval_prune_fixed(model, scores, syn_target):
    """Die 'feste' Regel aus dem budget-Skript: (1-score) * Ersparnis max."""
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(0.15 * o), 8) for o in orig]
    starts = model.hidden_start_indices
    while model.active_synapses > syn_target:
        counts = model.neuron_counts
        if all(c <= f for c, f in zip(counts, floor)):
            break
        cand = []
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue
            mask = model.neuron_masks[L].cpu().numpy()
            in_dim = 28 * 28 if L == 0 else counts[L - 1]
            out_dim = model.n_classes if L == model.n_layers - 1 else counts[L + 1]
            sav = in_dim + out_dim
            for nl in np.where(mask > 0)[0]:
                cand.append(((1.0 - scores[starts[L] + nl]) * sav, L, int(nl)))
        if not cand:
            break
        cand.sort(key=lambda t: -t[0])
        _, L, nl = cand[0]
        mask = model.neuron_masks[L].clone()
        mask[nl] = 0.0
        setattr(model, f"neuron_mask_{L}", mask)
    return model.neuron_counts


def eval_random(model, feats, syn_target):
    torch.manual_seed(0)
    orig = list(model.HIDDEN_SIZES)
    floor = [max(int(0.15 * o), 8) for o in orig]
    steps = 0
    while model.active_synapses > syn_target and steps < MAX_STEPS_PER_EPISODE:
        counts = model.neuron_counts
        if all(c <= f for c, f in zip(counts, floor)):
            break
        choices = []
        for L in range(model.n_layers):
            if counts[L] <= floor[L]:
                continue
            for nl in np.where(model.neuron_masks[L].cpu().numpy() > 0)[0]:
                choices.append((L, int(nl)))
        if not choices:
            break
        L, nl = choices[int(np.random.randint(len(choices)))]
        mask = model.neuron_masks[L].clone()
        mask[nl] = 0.0
        setattr(model, f"neuron_mask_{L}", mask)
        steps += 1
    return model.neuron_counts


def run_eval():
    policy = PrunerPolicy(n_feat=6).to(device)
    policy_path = os.path.join(_HERE, "models", "pruner_policy.pt")
    if not os.path.exists(policy_path):
        print(f"  Policy nicht gefunden: {policy_path}. Erst trainieren "
              f"(python structural_ml_rl_pruner.py ohne --eval).", flush=True)
        return
    policy.load_state_dict(torch.load(policy_path, map_location=device,
                                      weights_only=True))
    policy.eval()

    # Meta-Scorer fuer die feste Regel
    scorer_cands = [os.path.join(_HERE, "models", "neuron_scorer_meta.pt"),
                    os.path.join(_HERE, "..", "..", "Code", "models",
                                 "neuron_scorer_meta.pt"),
                    r"D:\Dynamic Neural Networt (DNN)\Code\models\neuron_scorer_meta.pt"]
    scorer_path = next((p for p in scorer_cands if os.path.exists(p)), scorer_cands[0])
    scorer = mt.NeuronScorer(n_feat=6).to(device)
    scorer.load_state_dict(torch.load(scorer_path, map_location=device,
                                      weights_only=True))
    scorer.eval()

    seeds = EPISODE_SEEDS
    rows = []
    for seed in seeds:
        acc0 = None
        # ---- RL deterministisch ----
        torch.manual_seed(seed); np.random.seed(seed)
        model_rl = mt.DynamicMLP(LAYOUT).to(device)
        train_epochs(model_rl, WARM)
        acc0 = mt.measure_accuracy(model_rl)
        feats_rl = collect_feats(model_rl)
        keep_rl = eval_deterministic(model_rl, policy, feats_rl, SYN_TARGET)
        k_rl = [torch.nonzero(m > 0).squeeze(1).cpu() for m in model_rl.neuron_masks]
        c_rl = compact_dynamicmlp(model_rl, keep_rl, k_rl)
        train_epochs(c_rl, FINETUNE)
        acc_rl = mt.measure_accuracy(c_rl)
        syn_rl = c_rl.active_synapses

        # ---- feste Regel ----
        torch.manual_seed(seed); np.random.seed(seed)
        model_f = mt.DynamicMLP(LAYOUT).to(device)
        train_epochs(model_f, WARM)
        feats_f = collect_feats(model_f)
        with torch.no_grad():
            s_f = scorer(torch.from_numpy(feats_f).float().to(device)).cpu().numpy()
        keep_f = eval_prune_fixed(model_f, s_f, SYN_TARGET)
        k_f = [torch.nonzero(m > 0).squeeze(1).cpu() for m in model_f.neuron_masks]
        c_f = compact_dynamicmlp(model_f, keep_f, k_f)
        train_epochs(c_f, FINETUNE)
        acc_f = mt.measure_accuracy(c_f)
        syn_f = c_f.active_synapses

        # ---- Random ----
        torch.manual_seed(seed); np.random.seed(seed)
        model_r = mt.DynamicMLP(LAYOUT).to(device)
        train_epochs(model_r, WARM)
        feats_r = collect_feats(model_r)
        keep_r = eval_random(model_r, feats_r, SYN_TARGET)
        k_r = [torch.nonzero(m > 0).squeeze(1).cpu() for m in model_r.neuron_masks]
        c_r = compact_dynamicmlp(model_r, keep_r, k_r)
        train_epochs(c_r, FINETUNE)
        acc_r = mt.measure_accuracy(c_r)
        syn_r = c_r.active_synapses

        rows.append({"seed": seed, "acc0": acc0,
                     "acc_rl": acc_rl, "syn_rl": syn_rl, "layers_rl": keep_rl,
                     "acc_fixed": acc_f, "syn_fixed": syn_f, "layers_fixed": keep_f,
                     "acc_random": acc_r, "syn_random": syn_r, "layers_random": keep_r})
        print(f"  seed={seed} acc_voll={acc0:.2f} | RL={acc_rl:.2f} "
              f"fixed={acc_f:.2f} random={acc_r:.2f}", flush=True)

    # Zusammenfassung
    print("\n" + "=" * 76)
    print("  AUSWERTUNG (Mittelwert): gelernte RL-Policy vs. feste Regel")
    print("=" * 76)
    for key, lbl in [("acc_rl", "RL  (gelernt)  "), ("acc_fixed", "fest (Score*s)"),
                     ("acc_random", "random        ")]:
        print(f"  {lbl}: {np.mean([r[key] for r in rows]):.2f}%")
    _save_eval_csv(rows)


def _save_eval_csv(rows):
    csvp = os.path.join(_HERE, "RESULT_rl_pruner_eval.csv")
    import ast
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "acc0", "acc_rl", "syn_rl", "layers_rl",
                    "acc_fixed", "syn_fixed", "layers_fixed",
                    "acc_random", "syn_random", "layers_random"])
        for r in rows:
            w.writerow([r["seed"], round(r["acc0"], 3),
                        round(r["acc_rl"], 3), r["syn_rl"], r["layers_rl"],
                        round(r["acc_fixed"], 3), r["syn_fixed"], r["layers_fixed"],
                        round(r["acc_random"], 3), r["syn_random"], r["layers_random"]])
    print(f"  CSV gespeichert: {csvp}", flush=True)


if __name__ == "__main__":
    if "--eval" in sys.argv:
        run_eval()
    else:
        train_rl()
