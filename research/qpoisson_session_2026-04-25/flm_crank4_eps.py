"""
Crank 4: fine-grained trim_eps sweep.
Crank 1 said F1 (ε=0.1) wins, F2 (ε=0.5) is partial.  Let's find the optimum.

trim_eps ∈ {0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0}
DGPs: D1 (Poisson d=4 n=20k), D2 (Poisson d=8 n=20k), D3 (Poisson d=4 n=50k).
M=30 reps each.  192 total runs.
"""

import numpy as np, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys; sys.path.insert(0, "/tmp/qpoisson_if")
from flm_robustness import (
    _build_truth, cross_fit_psi, poisson_if, summarize, cp_ci, N_WORKERS
)


EPS_GRID = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
DGPS = [("D1_P_d4_n20k", 4, 20_000),
        ("D2_P_d8_n20k", 8, 20_000),
        ("D3_P_d4_n50k", 4, 50_000)]
M = 25


def simulate(rng, n, d, truth):
    import torch
    x = rng.standard_normal((n, d)).astype(np.float32)
    t = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = truth["alpha"](xt).numpy()
        b = truth["beta"](xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return x, t, y, truth["mu_star"]


def run_one(payload):
    eps, dgp_name, d, n, mc_seed = payload
    truth = _build_truth(d, seed=1001)
    rng = np.random.default_rng(mc_seed)
    x, t, y, mu_star = simulate(rng, n, d, truth)
    psi, beta_hat, _, _, _ = cross_fit_psi(x, t, y, seed=mc_seed,
                                            trim_eps=eps, ensemble_r=1)
    mu_hat = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    return dict(eps=eps, dgp=dgp_name, mc_seed=mc_seed,
                mu_hat=mu_hat, se=se, mu_star=mu_star,
                covered=bool(abs(mu_hat - mu_star) <= 1.96 * se),
                mu_naive=float(beta_hat.mean()))


def main():
    payloads = []
    for dgp_name, d, n in DGPS:
        for eps in EPS_GRID:
            for m in range(M):
                payloads.append((eps, dgp_name, d, n, 12000 + m * 31))
    print(f"[crank4] runs: {len(payloads)}")
    t0 = time.time()
    by_group = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one, p) for p in payloads]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            by_group.setdefault((r["eps"], r["dgp"]), []).append(r)
            done += 1
            if done % 30 == 0:
                dt = time.time() - t0
                eta = dt * (len(futs) - done) / max(done, 1)
                print(f"  {done:4d}/{len(futs)}  t={dt:5.0f}s  eta={eta:.0f}s")
    print("\n=== CRANK 4: trim_eps sweep ===")
    for dgp_name, _, _ in DGPS:
        print(f"\n--- {dgp_name} ---")
        print(f"  {'eps':>6}{'bias':>10}{'|b|/SE':>9}{'SEratio':>9}"
              f"{'cov':>7}{'CP-CI':>14}{'verdict':>10}")
        for eps in EPS_GRID:
            res = by_group[(eps, dgp_name)]
            s = summarize(f"eps_{eps}", dgp_name, res)
            print(f"  {eps:>6.2f}{s['bias']:>+10.4f}"
                  f"{s['abs_bias_over_se']:>9.3f}"
                  f"{s['se_ratio']:>9.3f}"
                  f"{s['coverage']:>7.3f}"
                  f" [{s['cp_lo']:.2f},{s['cp_hi']:.2f}]"
                  f"{s['verdict']:>10}")

    print(f"\n[crank4 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
