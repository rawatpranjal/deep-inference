"""
Crank 3: DNN architecture stress test.

For the failing Poisson d=8 N=20k case, try every DNN architecture variant:
  A1: (32, 16) ReLU
  A2: (64, 32) ReLU  (baseline)
  A3: (128, 64) ReLU
  A4: (256, 128) ReLU
  A5: (64, 32) Tanh
  A6: (32, 32) Tanh  (matches truth architecture exactly)
  A7: (64, 32) ReLU + BatchNorm
  A8: (64, 32) ReLU + Dropout 0.1
  A9: (64, 32, 16) ReLU deep

All with best-fix IF wrapper (Λ-trim + winsorize).
Report which architecture gives best FML coverage/bias.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_robustness import (
    _build_truth, poisson_nll, poisson_if, winsorize_psi, summarize,
    K_FOLDS, DNN_LR, DNN_MAX_EPOCHS, DNN_PATIENCE, DNN_BATCH, cp_ci, N_WORKERS,
)


# ---------- configurable ThetaNet ----------
class FlexibleThetaNet(nn.Module):
    def __init__(self, d_in, hidden, activation="relu",
                 batchnorm=False, dropout=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_a = nn.Linear(prev, 1)
        self.head_b = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


def train_flex(net, x_tr, t_tr, y_tr, x_va, t_va, y_va):
    opt = optim.Adam(net.parameters(), lr=DNN_LR, weight_decay=1e-5)
    xtr, ttr, ytr = map(torch.from_numpy, (x_tr, t_tr, y_tr))
    xva, tva, yva = map(torch.from_numpy, (x_va, t_va, y_va))
    n = len(xtr)
    best = float("inf"); best_st = None; pat = 0
    for ep in range(DNN_MAX_EPOCHS):
        net.train()
        idx = np.random.permutation(n)
        for s in range(0, n, DNN_BATCH):
            sel = idx[s:s + DNN_BATCH]
            a, b = net(xtr[sel])
            mu = torch.exp(a + b * ttr[sel])
            loss = poisson_nll(mu, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            av, bv = net(xva)
            muv = torch.exp(av + bv * tva)
            val = poisson_nll(muv, yva).item()
        if val < best - 1e-5:
            best = val
            best_st = {k: v.detach().clone() for k, v in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= DNN_PATIENCE: break
    if best_st is not None:
        net.load_state_dict(best_st)
    return net


def pred(net, x):
    net.eval()
    with torch.no_grad():
        a, b = net(torch.from_numpy(x))
    return a.numpy(), b.numpy()


# ---------- architecture configs ----------
ARCHS = {
    "A1_relu_32_16":      dict(hidden=(32, 16), activation="relu"),
    "A2_relu_64_32":      dict(hidden=(64, 32), activation="relu"),
    "A3_relu_128_64":     dict(hidden=(128, 64), activation="relu"),
    "A4_relu_256_128":    dict(hidden=(256, 128), activation="relu"),
    "A5_tanh_64_32":      dict(hidden=(64, 32), activation="tanh"),
    "A6_tanh_32_32":      dict(hidden=(32, 32), activation="tanh"),
    "A7_relu_64_32_BN":   dict(hidden=(64, 32), activation="relu", batchnorm=True),
    "A8_relu_64_32_drop": dict(hidden=(64, 32), activation="relu", dropout=0.1),
    "A9_relu_deep":       dict(hidden=(64, 32, 16), activation="relu"),
}

DGPS = [("D1_P_d4_n20k", 4, 20_000), ("D2_P_d8_n20k", 8, 20_000)]
M_MC = 15
TRIM_EPS = 0.10
WINSOR = (0.01, 0.99)


def simulate(rng, n, d, truth):
    x = rng.standard_normal((n, d)).astype(np.float32)
    t = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = truth["alpha"](xt).numpy()
        b = truth["beta"](xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return x, t, y, a, b, mu, truth["mu_star"]


def run_worker(payload):
    arch_name, arch_cfg, dgp_name, d, n, mc_seed = payload
    rng = np.random.default_rng(mc_seed)
    truth = _build_truth(d, seed=1001)
    x, t, y, _, _, _, mu_star = simulate(rng, n, d, truth)

    folds = rng.integers(K_FOLDS, size=n)
    alpha_hat = np.zeros(n); beta_hat = np.zeros(n)
    for k in range(K_FOLDS):
        me = folds == k
        tr_all = np.where(~me)[0]
        perm = np.random.default_rng(mc_seed * 17 + k).permutation(len(tr_all))
        n_val = max(int(0.15 * len(tr_all)), 100)
        va_idx = tr_all[perm[:n_val]]
        tr_idx = tr_all[perm[n_val:]]
        torch.manual_seed(mc_seed * 100 + k)
        net = FlexibleThetaNet(d, **arch_cfg)
        net = train_flex(net,
                         x[tr_idx], t[tr_idx], y[tr_idx],
                         x[va_idx], t[va_idx], y[va_idx])
        a_e, b_e = pred(net, x[me])
        alpha_hat[me] = a_e; beta_hat[me] = b_e

    psi, _, _ = poisson_if(alpha_hat, beta_hat, t, y, trim_eps=TRIM_EPS)
    psi_w = winsorize_psi(psi, *WINSOR)
    mu_hat = float(psi_w.mean())
    se = float(psi_w.std(ddof=1) / np.sqrt(n))
    return dict(arch=arch_name, dgp=dgp_name, mc_seed=mc_seed,
                mu_hat=mu_hat, se=se, mu_star=mu_star,
                covered=bool(abs(mu_hat - mu_star) <= 1.96 * se),
                mu_naive=float(beta_hat.mean()))


def main():
    t0 = time.time()
    payloads = []
    for dgp_name, d, n in DGPS:
        for arch_name, arch_cfg in ARCHS.items():
            for m in range(M_MC):
                payloads.append((arch_name, arch_cfg, dgp_name, d, n,
                                 11_000 + m * 31))
    print(f"[crank3] runs: {len(payloads)}")
    t_start = time.time()
    by_group = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_worker, p) for p in payloads]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            by_group.setdefault((r["arch"], r["dgp"]), []).append(r)
            done += 1
            if done % 30 == 0:
                dt = time.time() - t_start
                eta = dt * (len(futs) - done) / max(done, 1)
                print(f"  {done:4d}/{len(futs)}  t={dt:5.0f}s  eta={eta:.0f}s")

    # report
    print("\n" + "=" * 100)
    print("CRANK 3: Architecture stress (Poisson, fix=Λ-trim+winsor)".center(100))
    print("=" * 100)
    for dgp_name, _, _ in DGPS:
        print(f"\n--- {dgp_name} ---")
        print(f"  {'arch':<22}{'bias':>9}{'|b|/SE':>8}{'SEratio':>9}"
              f"{'cov':>7}{'CP-CI':>14}{'verdict':>10}")
        rows = sorted([k for k in by_group if k[1] == dgp_name])
        for key in rows:
            s = summarize(key[0], dgp_name, by_group[key])
            print(f"  {s['fix']:<22}{s['bias']:>+9.4f}"
                  f"{s['abs_bias_over_se']:>8.3f}{s['se_ratio']:>9.3f}"
                  f"{s['coverage']:>7.3f} [{s['cp_lo']:.2f},{s['cp_hi']:.2f}]"
                  f"{s['verdict']:>10}")

    print(f"\n[crank3 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
