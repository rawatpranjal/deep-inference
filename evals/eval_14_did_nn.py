"""
eval_14_did_nn: heterogeneous neural 2x2 DiD (model='did', target='tau').

Validates the four mathematical objects behind the neural DiD estimator of E[tau(X)]:

  Test 1  Recovery  : theta_hat(X) vs theta*(X) = [alpha, gamma, lambda, tau]
  Test 2  Autodiff  : DiDModel closed-form score/Hessian vs torch autodiff
  Test 3  Lambda    : aggregate analytic Lambda_hat vs oracle E[WW']
  Test 4  Coverage  : Monte Carlo coverage of the 95% CI for E[tau(X)] (Regime B)

Plus a benchmark of neural E[tau(X)] vs the closed-form did_2x2() on a homogeneous DGP.

Run:
    python3 -m evals.eval_14_did_nn 2>&1 | tee evals/reports/eval_14_did_nn_$(date +%Y%m%d_%H%M%S).txt
    python3 -m evals.eval_14_did_nn --quick   # fast smoke (small M)
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import inference, did  # noqa: E402
from deep_inference.models import DiDModel  # noqa: E402
from deep_inference.lambda_.analytic import AnalyticLambda  # noqa: E402
from evals.dgp_did_nn import DiDNNDGP  # noqa: E402
from evals.common.metrics import (  # noqa: E402
    validate_coverage_run,
    validate_recovery_run,
    validate_autodiff_run,
    validate_lambda_run,
    format_validation_table,
)

QUICK = "--quick" in sys.argv

# Regime B (linear, 2-way split). Coverage needs adequate n/epochs (cf. eval_06 linear:
# n=5000, epochs=200 -> 96%); --quick uses a lean config only to smoke-test the pipeline.
if QUICK:
    M, N, N_FOLDS, EPOCHS, PATIENCE = 8, 2000, 5, 50, 15
else:
    M, N, N_FOLDS, EPOCHS, PATIENCE = 25, 5000, 5, 150, 40
HIDDEN = [32, 16]
LR = 0.01
N_JOBS = 4

DGP = DiDNNDGP()
MU_TRUE = DGP.mu_true()


def _fit_once(seed: int):
    """One coverage replication: fit neural DiD, return (beta_hat, se, covered, z)."""
    Y, G, P, X = DGP.generate(N, seed=seed)
    r = did(
        Y, G, P, X,
        hidden_dims=HIDDEN, n_folds=N_FOLDS, epochs=EPOCHS,
        lr=LR, patience=PATIENCE,
    )
    covered = (r.ci_lower <= MU_TRUE) and (MU_TRUE <= r.ci_upper)
    return r.mu_hat, r.se, bool(covered), (r.mu_hat - MU_TRUE) / r.se


def test_recovery():
    print("\n[Test 1] Parameter recovery: theta_hat(X) vs theta*(X)")
    Y, G, P, X = DGP.generate(N, seed=12345)
    r = did(Y, G, P, X, hidden_dims=HIDDEN, n_folds=N_FOLDS, epochs=EPOCHS,
                   lr=LR, patience=PATIENCE)
    theta_hat = r.theta_hat.numpy()
    theta_star = DGP.theta_star(X)
    names = ["alpha", "gamma", "lambda", "tau"]
    metrics = {}
    for j, nm in enumerate(names):
        rmse = float(np.sqrt(np.mean((theta_hat[:, j] - theta_star[:, j]) ** 2)))
        corr = float(np.corrcoef(theta_hat[:, j], theta_star[:, j])[0, 1])
        metrics[f"rmse_{nm}"] = rmse
        metrics[f"corr_{nm}"] = corr
        print(f"  {nm:7s}: RMSE={rmse:.4f}  corr={corr:.4f}")
    print(f"  E[tau_hat]={theta_hat[:,3].mean():.4f}  (true E[tau]={MU_TRUE:.4f})")
    passed, criteria = validate_recovery_run(metrics)
    print(format_validation_table(criteria))
    return passed


def test_autodiff():
    print("\n[Test 2] Autodiff: DiDModel score/Hessian vs torch autodiff")
    import torch
    model = DiDModel()
    rng = np.random.default_rng(0)
    max_g, max_h = 0.0, 0.0
    for _ in range(200):
        y = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)
        G, P = float(rng.integers(0, 2)), float(rng.integers(0, 2))
        t = torch.tensor([G, P, G * P], dtype=torch.float64)
        theta = torch.tensor(rng.standard_normal(4), dtype=torch.float64, requires_grad=True)
        loss = model.loss(y, t, theta)
        (g_ad,) = torch.autograd.grad(loss, theta, create_graph=True)
        h_ad = torch.stack([torch.autograd.grad(g_ad[j], theta, retain_graph=True)[0] for j in range(4)])
        max_g = max(max_g, float((model.score(y, t, theta.detach()) - g_ad.detach()).abs().max()))
        max_h = max(max_h, float((model.hessian(y, t, theta.detach()) - h_ad).abs().max()))
    print(f"  max gradient error: {max_g:.2e}")
    print(f"  max hessian  error: {max_h:.2e}")
    passed, criteria = validate_autodiff_run({"gradient_error": max_g, "hessian_error": max_h})
    print(format_validation_table(criteria))
    return passed


def test_lambda():
    print("\n[Test 3] Lambda: aggregate analytic Lambda_hat vs oracle E[WW']")
    import torch
    Y, G, P, X = DGP.generate(50000, seed=99)
    T = torch.tensor(np.column_stack([G, P, G * P]), dtype=torch.float32)
    lam = AnalyticLambda(method="aggregate")
    lam.fit(torch.tensor(X, dtype=torch.float32), T, torch.tensor(Y, dtype=torch.float32), None, DiDModel())
    L_hat = lam.predict(torch.tensor(X[:1], dtype=torch.float32))[0].numpy()
    L_oracle = DGP.oracle_lambda()
    frob = float(np.linalg.norm(L_hat - L_oracle) / np.linalg.norm(L_oracle))
    min_eig = float(np.linalg.eigvalsh(L_hat).min())
    non_psd = int(min_eig <= 0)
    print(f"  Frobenius rel error: {frob:.4f}")
    print(f"  min eigenvalue     : {min_eig:.4f}")
    print(f"  Lambda_hat:\n{np.round(L_hat,4)}")
    passed, criteria = validate_lambda_run(
        {"frobenius_error": frob, "min_eigenvalue": min_eig, "non_psd_count": non_psd}
    )
    print(format_validation_table(criteria))
    return passed


def test_coverage():
    print(f"\n[Test 4] Coverage: 95% CI for E[tau(X)] over M={M} reps (n={N}, Regime B)")
    t0 = time.time()
    try:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_fit_once)(m + 1) for m in range(M))
    except Exception:
        out = [_fit_once(m + 1) for m in range(M)]
    betas = np.array([o[0] for o in out])
    ses = np.array([o[1] for o in out])
    covered = np.array([o[2] for o in out])
    z = np.array([o[3] for o in out])

    coverage = float(covered.mean())
    emp_se = float(betas.std(ddof=1))
    mean_se = float(ses.mean())
    metrics = {
        "coverage": coverage,
        "se_ratio": emp_se / mean_se,
        "bias": float(betas.mean() - MU_TRUE),
        "z_mean": float(z.mean()),
        "z_std": float(z.std(ddof=1)),
    }
    print(f"  mean(beta_hat) : {betas.mean():.4f}   (true E[tau]={MU_TRUE:.4f})")
    print(f"  bias           : {metrics['bias']:+.4f}")
    print(f"  empirical SE   : {emp_se:.4f}")
    print(f"  mean est. SE   : {mean_se:.4f}")
    print(f"  SE ratio       : {metrics['se_ratio']:.4f}")
    print(f"  coverage       : {coverage*100:.1f}%   ({covered.sum()}/{M})")
    print(f"  z mean / std   : {metrics['z_mean']:+.4f} / {metrics['z_std']:.4f}")
    print(f"  elapsed        : {time.time()-t0:.1f}s")
    passed, criteria = validate_coverage_run(metrics)
    print(format_validation_table(criteria))
    return passed


def benchmark_homogeneous():
    print("\n[Benchmark] homogeneous DGP: neural E[tau(X)] vs closed-form did_2x2()")
    hom = DiDNNDGP(A1=0.0, G1=0.0, L1=0.0, T1=0.0)  # tau constant = T0
    Y, G, P, X = hom.generate(N, seed=2024)
    cf = did(Y, G, P)
    nn = did(Y, G, P, X, hidden_dims=HIDDEN, n_folds=N_FOLDS, epochs=EPOCHS,
                    lr=LR, patience=PATIENCE)
    print(f"  closed-form did_2x2 : beta={cf.mu_hat:.4f}  se={cf.se:.4f}")
    print(f"  neural   did(X=..) : beta={nn.mu_hat:.4f}  se={nn.se:.4f}")
    print(f"  true tau            : {hom.T0:.4f}")
    diff = abs(cf.mu_hat - nn.mu_hat)
    se = max(cf.se, nn.se)
    ok = diff < 3 * se
    print(f"  |closed-form - neural| = {diff:.4f}  ({diff/se:.2f} SE)  -> {'OK' if ok else 'CHECK'}")
    return ok


def main():
    print("=" * 78)
    print(f"eval_14_did_nn: heterogeneous neural 2x2 DiD {'(QUICK)' if QUICK else ''}")
    print("=" * 78)
    print(f"DGP: alpha,gamma,lambda,tau heterogeneous in X0; (G,P) ⟂ X; E[tau]={MU_TRUE}")
    print(f"Config: M={M}, n={N}, n_folds={N_FOLDS}, epochs={EPOCHS}, hidden={HIDDEN}")

    results = {
        "Recovery": test_recovery(),
        "Autodiff": test_autodiff(),
        "Lambda": test_lambda(),
        "Coverage": test_coverage(),
        "Benchmark (hom.)": benchmark_homogeneous(),
    }

    print("\n" + "=" * 78)
    print("SCORECARD")
    print("-" * 78)
    for name, ok in results.items():
        print(f"  {name:18s}: {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print("-" * 78)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
