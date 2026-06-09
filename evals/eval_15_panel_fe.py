"""
eval_15_panel_fe: two-way fixed-effects panel DiD (model='did_fe', did_panel_fe).

Validates the within-transformed FE neural DiD estimator of E[tau(X)]:

  Test 1  Recovery  : tau_hat(X) vs tau*(X)
  Test 2  Autodiff  : FEPanelDiDModel closed-form score/Hessian vs torch autodiff
  Test 3  Lambda    : intercept-free analytic Lambda_hat = E[Dtilde^2|X] vs oracle (1x1)
  Test 4  Coverage  : 95% CI for E[tau(X)] over M reps, CONTINUOUS and BINARY (LPM)

Run:
    python3 -m evals.eval_15_panel_fe 2>&1 | tee evals/reports/eval_15_panel_fe_$(date +%Y%m%d_%H%M%S).txt
    python3 -m evals.eval_15_panel_fe --quick
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import did  # noqa: E402
from deep_inference.models import FEPanelDiDModel  # noqa: E402
from deep_inference.lambda_.analytic import AnalyticLambda  # noqa: E402
from deep_inference.utils import residualize_fixed_effects  # noqa: E402
from evals.dgp_panel_fe import PanelFEDGP  # noqa: E402
from evals.common.metrics import (  # noqa: E402
    validate_coverage_run,
    validate_recovery_run,
    validate_autodiff_run,
    validate_lambda_run,
    format_validation_table,
)

QUICK = "--quick" in sys.argv
M = 8 if QUICK else 50
N_FOLDS = 5
EPOCHS = 150
PATIENCE = 40
HIDDEN = [32, 16]
LR = 0.01
N_JOBS = 4

DGP = PanelFEDGP()
MU_TRUE_C = DGP.mu_true_continuous()
MU_TRUE_B = DGP.mu_true_binary()


def _fit(gen, mu_true, seed):
    Y, D, X, unit, time_, tau = gen(seed)
    r = did(Y, D=D, X=X, unit=unit, time=time_, hidden_dims=HIDDEN, n_folds=N_FOLDS,
                     epochs=EPOCHS, lr=LR, patience=PATIENCE)
    covered = (r.ci_lower <= mu_true) and (mu_true <= r.ci_upper)
    return r.mu_hat, r.se, bool(covered), (r.mu_hat - mu_true) / r.se


def test_recovery():
    print("\n[Test 1] Recovery: tau_hat(X) vs tau*(X) (continuous)")
    Y, D, X, unit, time_, tau = DGP.generate_continuous(seed=777)
    r = did(Y, D=D, X=X, unit=unit, time=time_, hidden_dims=HIDDEN, n_folds=N_FOLDS,
                     epochs=EPOCHS, lr=LR, patience=PATIENCE)
    tau_hat = r.theta_hat.numpy()[:, 0]
    rmse = float(np.sqrt(np.mean((tau_hat - tau) ** 2)))
    corr = float(np.corrcoef(tau_hat, tau)[0, 1])
    print(f"  tau: RMSE={rmse:.4f}  corr={corr:.4f}  E[tau_hat]={tau_hat.mean():.4f} (true {MU_TRUE_C})")
    passed, criteria = validate_recovery_run({"rmse_tau": rmse, "corr_tau": corr})
    print(format_validation_table(criteria))
    return passed


def test_autodiff():
    print("\n[Test 2] Autodiff: FEPanelDiDModel score/Hessian vs torch autodiff")
    import torch
    model = FEPanelDiDModel()
    rng = np.random.default_rng(0)
    max_g, max_h = 0.0, 0.0
    for _ in range(200):
        y = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)
        t = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)
        theta = torch.tensor(rng.standard_normal(1), dtype=torch.float64, requires_grad=True)
        loss = model.loss(y, t, theta)
        (g,) = torch.autograd.grad(loss, theta, create_graph=True)
        (h,) = torch.autograd.grad(g[0], theta)
        max_g = max(max_g, float((model.score(y, t, theta.detach()) - g.detach()).abs().max()))
        max_h = max(max_h, float((model.hessian(y, t, theta.detach())[0, 0] - h[0]).abs().max()))
    print(f"  max gradient error: {max_g:.2e}   max hessian error: {max_h:.2e}")
    passed, criteria = validate_autodiff_run({"gradient_error": max_g, "hessian_error": max_h})
    print(format_validation_table(criteria))
    return passed


def test_lambda():
    print("\n[Test 3] Lambda: intercept-free analytic E[Dtilde^2|X] vs oracle (1x1)")
    import torch
    # build one large-ish panel, residualize D, fit intercept-free analytic Lambda
    Y, D, X, unit, time_, tau = DGP.generate_continuous(seed=5)
    Dt = residualize_fixed_effects(D, unit, time_)
    lam = AnalyticLambda(method="aggregate", intercept=False)
    lam.fit(torch.tensor(X, dtype=torch.float32),
            torch.tensor(Dt, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32), None, FEPanelDiDModel())
    L_hat = float(lam.predict(torch.tensor(X[:1], dtype=torch.float32))[0, 0, 0])
    L_oracle = DGP.oracle_lambda()
    frob = abs(L_hat - L_oracle) / abs(L_oracle)
    print(f"  Lambda_hat={L_hat:.4f}  oracle={L_oracle:.4f}  rel err={frob:.4f}")
    passed, criteria = validate_lambda_run(
        {"frobenius_error": frob, "min_eigenvalue": L_hat, "non_psd_count": int(L_hat <= 0)}
    )
    print(format_validation_table(criteria))
    return passed


def _coverage(label, gen, mu_true):
    print(f"\n[Test 4-{label}] Coverage of E[tau(X)] over M={M} reps ({label}, true={mu_true})")
    t0 = time.time()
    try:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_fit)(gen, mu_true, m + 1) for m in range(M))
    except Exception:
        out = [_fit(gen, mu_true, m + 1) for m in range(M)]
    betas = np.array([o[0] for o in out]); ses = np.array([o[1] for o in out])
    covered = np.array([o[2] for o in out]); z = np.array([o[3] for o in out])
    metrics = {
        "coverage": float(covered.mean()),
        "se_ratio": float(betas.std(ddof=1) / ses.mean()),
        "bias": float(betas.mean() - mu_true),
        "z_mean": float(z.mean()),
        "z_std": float(z.std(ddof=1)),
    }
    print(f"  mean(beta)={betas.mean():.4f} bias={metrics['bias']:+.4f} "
          f"empSE={betas.std(ddof=1):.4f} estSE={ses.mean():.4f} ratio={metrics['se_ratio']:.3f}")
    print(f"  coverage={metrics['coverage']*100:.1f}% ({covered.sum()}/{M})  "
          f"z={metrics['z_mean']:+.3f}/{metrics['z_std']:.3f}  ({time.time()-t0:.1f}s)")
    passed, criteria = validate_coverage_run(metrics)
    print(format_validation_table(criteria))
    return passed


def main():
    print("=" * 78)
    print(f"eval_15_panel_fe: two-way FE panel DiD {'(QUICK)' if QUICK else ''}")
    print("=" * 78)
    print(f"DGP: N={DGP.N} units x T={DGP.T} periods; E[tau] cont={MU_TRUE_C} bin={MU_TRUE_B}")
    print(f"Config: M={M}, n_folds={N_FOLDS}, epochs={EPOCHS}, hidden={HIDDEN}")

    results = {
        "Recovery": test_recovery(),
        "Autodiff": test_autodiff(),
        "Lambda": test_lambda(),
        "Coverage (continuous)": _coverage("continuous", DGP.generate_continuous, MU_TRUE_C),
        "Coverage (binary/LPM)": _coverage("binary", DGP.generate_binary, MU_TRUE_B),
    }

    print("\n" + "=" * 78 + "\nSCORECARD\n" + "-" * 78)
    for name, ok in results.items():
        print(f"  {name:24s}: {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print("-" * 78 + f"\nOVERALL: {'PASS' if overall else 'FAIL'}\n" + "=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
