"""
sim_04_did: simulation study for the difference-in-differences estimator family.

Reports parameter recovery and 95% CI coverage of the average DiD effect for:
  Block A  Neural 2x2 saturated DiD          did_2x2_nn   target E[tau(X)]
  Block B  Two-way FE panel DiD, continuous  did_panel_fe target E[tau(X)]
  Block C  Two-way FE panel DiD, binary/LPM  did_panel_fe target E[tau(X)]

Reuses the oracle DGPs from the evals (evals.dgp_did_nn, evals.dgp_panel_fe) and the
shared coverage reporting in simulations.common. Recovery is the RMSE / correlation of
the estimated heterogeneous effect tau_hat(X) vs the truth on a held representative
sample; coverage is over M Monte Carlo replications.

Run (heavy; background recommended):
    python3 -m simulations.sim_04_did 2>&1 | tee evals/reports/sim_04_did_$(date +%Y%m%d_%H%M%S).txt
    python3 -m simulations.sim_04_did --quick
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import did_2x2_nn, did_panel_fe  # noqa: E402
from evals.dgp_did_nn import DiDNNDGP  # noqa: E402
from evals.dgp_panel_fe import PanelFEDGP  # noqa: E402
from simulations.common import compute_coverage_metrics, print_coverage_report  # noqa: E402

QUICK = "--quick" in sys.argv
M = 8 if QUICK else 30
N_JOBS = 4


def _recovery(tau_hat, tau_star):
    rmse = float(np.sqrt(np.mean((tau_hat - tau_star) ** 2)))
    corr = float(np.corrcoef(tau_hat, tau_star)[0, 1])
    return rmse, corr


# ── Block A: neural 2x2 saturated DiD ──
def block_neural_2x2():
    dgp = DiDNNDGP()
    mu_true = dgp.mu_true()
    n, folds, epochs = (2000, 3, 60) if QUICK else (3000, 5, 120)

    def one(seed):
        Y, G, P, X = dgp.generate(n, seed=seed)
        r = did_2x2_nn(Y, G, P, X, n_folds=folds, epochs=epochs, hidden_dims=[32, 16], patience=30)
        tau_hat = r.theta_hat.numpy()[:, 3]
        tau_star = dgp.theta_star(X)[:, 3]
        return r.mu_hat, r.se, _recovery(tau_hat, tau_star)

    return _run_block("BLOCK A: Neural 2x2 saturated DiD", one, mu_true)


# ── Blocks B/C: FE panel DiD ──
def block_fe(label, binary):
    dgp = PanelFEDGP()
    mu_true = dgp.mu_true_binary() if binary else dgp.mu_true_continuous()
    folds, epochs = (3, 60) if QUICK else (5, 120)
    gen = dgp.generate_binary if binary else dgp.generate_continuous

    def one(seed):
        Y, D, X, unit, time_, tau_star = gen(seed)
        r = did_panel_fe(Y, D, X, unit, time_, n_folds=folds, epochs=epochs,
                         hidden_dims=[32, 16], patience=30)
        tau_hat = r.theta_hat.numpy()[:, 0]
        return r.mu_hat, r.se, _recovery(tau_hat, tau_star)

    return _run_block(label, one, mu_true)


def _run_block(name, one, mu_true):
    t0 = time.time()
    try:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(m + 1) for m in range(M))
    except Exception:
        out = [one(m + 1) for m in range(M)]
    mu_hats = [o[0] for o in out]
    ses = [o[1] for o in out]
    rmses = [o[2][0] for o in out]
    corrs = [o[2][1] for o in out]
    metrics = compute_coverage_metrics(mu_hats, ses, mu_true)
    report = print_coverage_report(metrics, name, mu_true)
    rec = (f"  Recovery tau(X): RMSE={np.mean(rmses):.4f}  corr={np.mean(corrs):.4f}  "
           f"(mean over {M} reps)   elapsed={time.time()-t0:.1f}s")
    print(rec)
    return name, metrics, report + "\n" + rec


def main():
    print("=" * 60)
    print(f"  SIM 4: DIFFERENCE-IN-DIFFERENCES FAMILY {'(QUICK)' if QUICK else ''}")
    print(f"  M = {M} replications per block")
    print("=" * 60)

    blocks = [
        block_neural_2x2(),
        block_fe("BLOCK B: FE panel DiD (continuous)", binary=False),
        block_fe("BLOCK C: FE panel DiD (binary / LPM)", binary=True),
    ]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, m, _ in blocks:
        print(f"  {name}")
        print(f"      coverage={m['coverage']*100:.1f}%  se_ratio={m['se_ratio']:.3f}  "
              f"|bias|={abs(m['bias']):.4f}  z_mean={m['z_mean']:+.3f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
