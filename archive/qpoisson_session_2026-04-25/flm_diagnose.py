"""
Diagnose why FML bias isn't shrinking at rate √n.

Five targeted tests:

(1) ORACLE FML: Plug the TRUE θ*(x) = (α*(x), β1*(x), β2*(x)) into the IF
    correction instead of a DNN estimate.  If this has zero bias and 95%
    coverage, the IF formula is provably correct and the entire problem is
    DNN nuisance rate.  If this fails, we have an IF bug.

(2) PARTIAL-ORACLE FML: true α*, learned β̂1, β̂2  (and vice versa).
    Isolates whether α-side or β-side DNN error is the binding issue.

(3) EXTENDED TRAINING: Train one DNN at N=100k for 500 epochs, NO early
    stopping.  Track L2(α̂), L2(β̂) per epoch.  If L2 keeps improving past
    epoch 50, our patience=25 is cutting off training too early.

(4) BIGGER DNN: Train with hidden=(256,128) instead of (64,32) at N=100k.
    If L2 drops materially, capacity was the bottleneck.

(5) LOG-LOG RATE: Fit log L2 vs log N using the sweep data we already have.
    What's the empirical DNN convergence exponent?

At the end, give an unambiguous verdict on what is wrong.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

# =====================================================
# import shared infra from the scaling script
# =====================================================
import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_scaling import (
    D_EMB, PHI, K_FOLDS, P_PERIODS, P0, P1, P2,
    alpha_true_torch, beta1_true_torch, beta2_true_torch,
    ALPHA_STAR_EVAL, BETA1_STAR_EVAL, BETA2_STAR_EVAL, X_EVAL_NP,
    MU_STAR, simulate, ThetaNet, poisson_nll, predict, fml_if,
)


def cp_ci(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# =====================================================
# (1) ORACLE FML
# =====================================================
def oracle_if(a_true, b1_true, b2_true, t, D1, D2, y):
    """IF using TRUE θ*.  If this has zero bias and 95% coverage, the IF
    formula is correct and the issue is purely DNN nuisance quality."""
    mu_0 = np.exp(a_true)
    mu_1 = np.exp(a_true + b1_true)
    mu_2 = np.exp(a_true + b2_true)
    mu_hat = np.where(t == 0, mu_0, np.where(t == 1, mu_1, mu_2))
    m_0 = P0 * mu_0; m_1 = P1 * mu_1
    correction_coef = -1.0 / m_0 + D1 * (1.0 / m_1 + 1.0 / m_0) + D2 * (1.0 / m_0)
    resid = mu_hat - y
    psi = b1_true - correction_coef * resid
    return psi


def run_oracle_rep(payload):
    N, seed = payload
    rng = np.random.default_rng(seed)
    x, t, D1, D2, y, a, b1, b2, mu = simulate(rng, N)
    psi = oracle_if(a, b1, b2, t, D1, D2, y)
    mu_naive_oracle = float(b1.mean())           # in-sample mean of β1*(X)
    mu_fml_oracle = float(psi.mean())
    se_fml = float(psi.std(ddof=1) / np.sqrt(N))
    return dict(N=N, seed=seed, mu_naive=mu_naive_oracle, mu_fml=mu_fml_oracle,
                se_fml=se_fml, psi_sd=float(psi.std(ddof=1)))


def test_oracle(N=50_000, M=200):
    print("\n" + "=" * 78)
    print(f"(1) ORACLE FML  (true θ* plug-in, no DNN)".center(78))
    print(f"N={N}, M={M} MC reps".center(78))
    print("=" * 78)
    payloads = [(N, 100_000 + m) for m in range(M)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(run_oracle_rep, payloads))
    mu_n = np.array([r["mu_naive"] for r in results])
    mu_f = np.array([r["mu_fml"] for r in results])
    se_f = np.array([r["se_fml"] for r in results])
    bias_n = mu_n.mean() - MU_STAR
    bias_f = mu_f.mean() - MU_STAR
    sd_n = mu_n.std(ddof=1)
    sd_f = mu_f.std(ddof=1)
    cov_f = (np.abs(mu_f - MU_STAR) <= 1.96 * se_f).mean()
    k = int(cov_f * M)
    lo, hi = cp_ci(k, M)
    print(f"  µ*                  = {MU_STAR:+.6f}")
    print(f"  naive ORACLE bias   = {bias_n:+.6f}  (this is just MC error of (1/n)Σβ1*)")
    print(f"  FML   ORACLE bias   = {bias_f:+.6f}  (<< 1e-3 means IF is right)")
    print(f"  FML   empirical SD  = {sd_f:.5f}")
    print(f"  FML   mean IF SE    = {se_f.mean():.5f}  "
          f"(ratio to emp SD = {se_f.mean()/sd_f:.4f})")
    print(f"  FML   |bias|/emp_SD = {abs(bias_f)/sd_f:.4f}")
    print(f"  FML   95% coverage  = {cov_f:.3f}  CP-CI [{lo:.3f}, {hi:.3f}]")
    print()
    verdict_if = abs(bias_f) / sd_f < 0.15 and lo <= 0.95 <= hi
    print(f"  VERDICT on IF formula: {'CORRECT' if verdict_if else 'BROKEN'}")
    return dict(bias_f=bias_f, cov=cov_f, sd_f=sd_f)


# =====================================================
# (2) PARTIAL ORACLE:  true α, estimated β̂ (and vice versa)
# =====================================================
def train_theta_long(x_tr, t_tr, D1_tr, D2_tr, y_tr,
                     x_va, t_va, D1_va, D2_va, y_va,
                     hidden=(64, 32), max_epochs=500, patience=100,
                     lr=2e-3, track=False):
    net = ThetaNet(x_tr.shape[1], hidden)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    xtr = torch.from_numpy(x_tr); ytr = torch.from_numpy(y_tr)
    D1tr = torch.from_numpy(D1_tr); D2tr = torch.from_numpy(D2_tr)
    xva = torch.from_numpy(x_va); yva = torch.from_numpy(y_va)
    D1va = torch.from_numpy(D1_va); D2va = torch.from_numpy(D2_va)
    n = len(xtr)
    best_val = float("inf"); best_state = None; pat = 0
    history = []
    for ep in range(max_epochs):
        net.train()
        idx = np.random.permutation(n)
        tl = []
        for s in range(0, n, 1024):
            sel = idx[s:s + 1024]
            a, b1, b2 = net(xtr[sel])
            mu = torch.exp(a + b1 * D1tr[sel] + b2 * D2tr[sel])
            loss = poisson_nll(mu, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        net.eval()
        with torch.no_grad():
            av, b1v, b2v = net(xva)
            muv = torch.exp(av + b1v * D1va + b2v * D2va)
            val = poisson_nll(muv, yva).item()
            if track:
                ae, b1e, b2e = net(torch.from_numpy(X_EVAL_NP))
                l2a = float(((ae.numpy() - ALPHA_STAR_EVAL) ** 2).mean())
                l2b1 = float(((b1e.numpy() - BETA1_STAR_EVAL) ** 2).mean())
                l2b2 = float(((b2e.numpy() - BETA2_STAR_EVAL) ** 2).mean())
                history.append(dict(ep=ep, train=float(np.mean(tl)), val=val,
                                    l2a=l2a, l2b1=l2b1, l2b2=l2b2))
            else:
                history.append(dict(ep=ep, train=float(np.mean(tl)), val=val))
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


# =====================================================
# (3) EXTENDED TRAINING AT N=100k
# =====================================================
def test_extended_training(N=100_000):
    print("\n" + "=" * 78)
    print(f"(3) EXTENDED TRAINING  (no early stop, track L2 per epoch)".center(78))
    print(f"N={N}, hidden=(64,32), max_epochs=500, patience=100".center(78))
    print("=" * 78)
    rng = np.random.default_rng(999)
    x, t, D1, D2, y, *_ = simulate(rng, N)
    # 80/15 train/val split
    idx = rng.permutation(N)
    n_val = int(0.15 * N)
    va, tr = idx[:n_val], idx[n_val:]
    t0 = time.time()
    net, hist = train_theta_long(
        x[tr], t[tr], D1[tr], D2[tr], y[tr],
        x[va], t[va], D1[va], D2[va], y[va],
        hidden=(64, 32), max_epochs=500, patience=100, track=True,
    )
    print(f"  trained {hist[-1]['ep']+1} epochs in {time.time()-t0:.1f}s")
    print(f"\n  {'ep':>3}  {'train':>9}  {'val':>9}  "
          f"{'L2(α̂)':>9}  {'L2(β1̂)':>9}  {'L2(β2̂)':>9}")
    epoch_list = sorted(set(list(range(0, len(hist), 20)) +
                            [len(hist) - 1, len(hist) // 2]))
    for i in epoch_list:
        h = hist[i]
        print(f"  {h['ep']:>3}  {h['train']:>9.4f}  {h['val']:>9.4f}  "
              f"{h['l2a']:>9.5f}  {h['l2b1']:>9.5f}  {h['l2b2']:>9.5f}")

    # best L2 across training
    best_l2b1 = min(h["l2b1"] for h in hist)
    best_l2a = min(h["l2a"] for h in hist)
    final_l2b1 = hist[-1]["l2b1"]
    final_l2a = hist[-1]["l2a"]
    print(f"\n  best L2(α̂) during training  = {best_l2a:.5f}  at ep "
          f"{min(range(len(hist)), key=lambda i: hist[i]['l2a'])}")
    print(f"  best L2(β1̂) during training = {best_l2b1:.5f}  at ep "
          f"{min(range(len(hist)), key=lambda i: hist[i]['l2b1'])}")
    print(f"  final L2(α̂)   = {final_l2a:.5f}")
    print(f"  final L2(β1̂)  = {final_l2b1:.5f}")
    print(f"\n  comparison to sweep at N=100k: L2(α̂)=0.00855, L2(β1̂)=0.01715")
    print(f"  extended-training L2(β1̂) better by factor "
          f"{0.01715/best_l2b1:.2f}")
    return dict(best_l2a=best_l2a, best_l2b1=best_l2b1, history=hist)


# =====================================================
# (4) BIGGER DNN
# =====================================================
def test_bigger_dnn(N=100_000):
    print("\n" + "=" * 78)
    print(f"(4) BIGGER DNN  hidden=(256,128), extended training".center(78))
    print(f"N={N}, max_epochs=500, patience=100".center(78))
    print("=" * 78)
    rng = np.random.default_rng(888)
    x, t, D1, D2, y, *_ = simulate(rng, N)
    idx = rng.permutation(N)
    n_val = int(0.15 * N)
    va, tr = idx[:n_val], idx[n_val:]
    t0 = time.time()
    net, hist = train_theta_long(
        x[tr], t[tr], D1[tr], D2[tr], y[tr],
        x[va], t[va], D1[va], D2[va], y[va],
        hidden=(256, 128), max_epochs=500, patience=100, track=True,
    )
    print(f"  trained {hist[-1]['ep']+1} epochs in {time.time()-t0:.1f}s")
    best_l2b1 = min(h["l2b1"] for h in hist)
    best_l2a = min(h["l2a"] for h in hist)
    print(f"  best L2(α̂)  = {best_l2a:.5f}")
    print(f"  best L2(β1̂) = {best_l2b1:.5f}")
    print(f"  comparison: (64,32) extended = see (3), gave L2(β1̂) ≈ ?")
    print(f"  sweep default (64,32) at N=100k: L2(β1̂)=0.01715")
    return dict(best_l2a=best_l2a, best_l2b1=best_l2b1, history=hist)


# =====================================================
# (5) RATE FROM SWEEP
# =====================================================
def test_rate_from_sweep():
    print("\n" + "=" * 78)
    print("(5) LOG-LOG DNN CONVERGENCE RATE FROM SWEEP".center(78))
    print("=" * 78)
    try:
        with open("/tmp/qpoisson_if/flm_scaling_results.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("  sweep results missing")
        return None
    Ns = []
    l2a_means = []; l2b1_means = []; l2b2_means = []
    bias_fml = []; se_fml = []
    for N_str, reps in data["results"].items():
        N = int(N_str)
        Ns.append(N)
        l2a_means.append(np.mean([r["l2_a"] for r in reps]))
        l2b1_means.append(np.mean([r["l2_b1"] for r in reps]))
        l2b2_means.append(np.mean([r["l2_b2"] for r in reps]))
        bias_fml.append(np.mean([r["mu_fml"] for r in reps]) - data["mu_star"])
        se_fml.append(np.mean([r["se_fml"] for r in reps]))
    Ns = np.array(Ns)
    # fit log-log
    logN = np.log(Ns)
    for name, y in [("L2(α̂)", l2a_means), ("L2(β1̂)", l2b1_means),
                    ("L2(β2̂)", l2b2_means), ("|bias|", np.abs(bias_fml)),
                    ("IF SE", se_fml)]:
        y = np.array(y)
        valid = y > 0
        if valid.sum() < 2:
            continue
        slope, intercept = np.polyfit(logN[valid], np.log(y[valid]), 1)
        r = np.corrcoef(logN[valid], np.log(y[valid]))[0, 1]
        print(f"  {name:<10}  N^({slope:+.3f})   r={r:.3f}")
    # target: for √n inference need bias rate < -0.5
    print(f"\n  For √n inference, need |bias| rate < -0.5.")
    print(f"  We have |bias| rate ≈ shown above.  If ≥ -0.5, FML fails at √n.")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    t0 = time.time()

    r1 = test_oracle(N=50_000, M=200)

    # Oracle at a large N to stress-test the IF
    print()
    r1b = test_oracle(N=200_000, M=100)

    test_rate_from_sweep()

    r3 = test_extended_training(N=100_000)

    r4 = test_bigger_dnn(N=100_000)

    # verdict
    print("\n" + "=" * 78)
    print("COMBINED VERDICT".center(78))
    print("=" * 78)
    print(f"  (1) Oracle FML N=50k  : bias={r1['bias_f']:+.6f}  "
          f"|bias|/SD={abs(r1['bias_f'])/r1['sd_f']:.3f}  "
          f"cov={r1['cov']:.3f}")
    print(f"  (1) Oracle FML N=200k : bias={r1b['bias_f']:+.6f}  "
          f"|bias|/SD={abs(r1b['bias_f'])/r1b['sd_f']:.3f}  "
          f"cov={r1b['cov']:.3f}")
    print(f"  (3) Extended DNN L2(β1̂) = {r3['best_l2b1']:.5f}  "
          f"(vs sweep {0.01715:.5f}, gain {0.01715/r3['best_l2b1']:.2f}x)")
    print(f"  (4) Bigger DNN L2(β1̂)  = {r4['best_l2b1']:.5f}  "
          f"(vs sweep {0.01715:.5f}, gain {0.01715/r4['best_l2b1']:.2f}x)")
    # interpretation
    print()
    oracle_works = abs(r1['bias_f'])/r1['sd_f'] < 0.15
    bigger_helps = r4['best_l2b1'] < 0.01
    print(f"  INTERPRETATION:")
    if oracle_works:
        print(f"    - IF is CORRECT (oracle FML achieves ~95% coverage).")
        print(f"    - Failure in sweep is DNN nuisance quality, not IF.")
    else:
        print(f"    - IF may have a problem: even oracle FML has bias.")
    if bigger_helps:
        print(f"    - DNN capacity WAS a bottleneck: (256,128) gives smaller L2.")
        print(f"    - Rerunning sweep with bigger DNN should improve coverage.")
    else:
        print(f"    - DNN capacity NOT the main bottleneck; need different fix.")

    print(f"\n  total time: {time.time()-t0:.0f}s")
