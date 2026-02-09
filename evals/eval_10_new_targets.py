"""
Eval 10: New Target Functionals

Validates DoseResponse, Profit, TailProbability, ConditionalVariance
against analytical oracles using the canonical logit DGP from eval_06.

DGP:
    X ~ Uniform(-2, 2)
    α*(x) = 0.5·sin(x), β*(x) = 1.0 + 0.5·x
    T = β*(x) + N(0, 0.5²)  [confounded]
    Y ~ Bernoulli(σ(α*(x) + β*(x)·T))

Oracle targets at t̃=0:
    DoseResponse:       E[σ(α*(X))]
    Profit:             E[0 · σ(α*(X))] = 0  (trivial at t̃=0, use t̃=1 instead)
    TailProbability:    E[σ(α*(X))]  (same as DoseResponse for logit)
    ConditionalVariance: E[σ(α*(X))·(1-σ(α*(X)))]

We use t̃=1.0 for Profit to get a non-trivial oracle.

Part 1: Jacobian validation (autodiff vs closed-form)
Part 2: Coverage validation (M=50 MC simulations)

NOTE ON t̃=0 JACOBIAN DEGENERACY:
    At t̃=0, the Jacobian H_θ = [∂h/∂α, ∂h/∂β] has zero β-component for
    DoseResponse and ConditionalVariance because ∂G/∂β = (∂G/∂η)·t̃ = 0.
    This means the influence function correction cannot address β estimation
    errors — they can only enter weakly through off-diagonal Λ terms.
    The result is SE underestimation (SE_ratio ≈ 2.0) and poor coverage at
    small n. At n≥8000 the uncorrected bias shrinks enough for valid coverage.
    Profit at t̃=1.0 has full-rank Jacobian and passes easily at n=3000.
    We also test DoseResponse at t̃=0.5 as a non-degenerate evaluation point.
"""

import sys
import numpy as np
import torch
from scipy.special import expit
from scipy import integrate
from dataclasses import dataclass
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import multiprocessing

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import CanonicalDGP, generate_canonical_dgp
from evals.common.metrics import validate_coverage_run, format_validation_table


# ═══════════════════════════════════════════════════════════════════
# ORACLE FORMULAS
# ═══════════════════════════════════════════════════════════════════

def oracle_dose_response(dgp: CanonicalDGP, t_tilde: float = 0.0) -> float:
    """E[σ(α*(X) + β*(X)·t̃)] via numerical integration."""
    def integrand(x):
        alpha = dgp.A0 + dgp.A1 * np.sin(x)
        beta = dgp.B0 + dgp.B1 * x
        p = expit(alpha + beta * t_tilde)
        return p * 0.25  # PDF of Uniform(-2, 2)
    result, _ = integrate.quad(integrand, dgp.X_low, dgp.X_high)
    return result


def oracle_profit(dgp: CanonicalDGP, t_tilde: float = 1.0) -> float:
    """E[t̃ · σ(α*(X) + β*(X)·t̃)] via numerical integration."""
    def integrand(x):
        alpha = dgp.A0 + dgp.A1 * np.sin(x)
        beta = dgp.B0 + dgp.B1 * x
        p = expit(alpha + beta * t_tilde)
        return t_tilde * p * 0.25
    result, _ = integrate.quad(integrand, dgp.X_low, dgp.X_high)
    return result


def oracle_tail_prob(dgp: CanonicalDGP, t_tilde: float = 0.0) -> float:
    """E[σ(α*(X) + β*(X)·t̃)] for logit (same as dose response)."""
    return oracle_dose_response(dgp, t_tilde)


def oracle_cond_var(dgp: CanonicalDGP, t_tilde: float = 0.0) -> float:
    """E[σ(α*(X) + β*(X)·t̃)·(1-σ(α*(X) + β*(X)·t̃))]."""
    def integrand(x):
        alpha = dgp.A0 + dgp.A1 * np.sin(x)
        beta = dgp.B0 + dgp.B1 * x
        p = expit(alpha + beta * t_tilde)
        return p * (1 - p) * 0.25
    result, _ = integrate.quad(integrand, dgp.X_low, dgp.X_high)
    return result


# ═══════════════════════════════════════════════════════════════════
# ORACLE JACOBIANS (for Part 1 validation)
# ═══════════════════════════════════════════════════════════════════

def oracle_jac_dose_response(theta: np.ndarray, t_tilde: float = 0.0) -> np.ndarray:
    """∂σ(η)/∂θ = p(1-p)·[1, t̃] where η = α + β·t̃."""
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t_tilde)
    dp = p * (1 - p)
    return np.array([dp, dp * t_tilde])


def oracle_jac_profit(theta: np.ndarray, t_tilde: float = 1.0) -> np.ndarray:
    """∂[t̃·σ(η)]/∂θ = t̃·p(1-p)·[1, t̃]."""
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t_tilde)
    dp = p * (1 - p)
    return np.array([t_tilde * dp, t_tilde * dp * t_tilde])


def oracle_jac_cond_var(theta: np.ndarray, t_tilde: float = 0.0) -> np.ndarray:
    """∂[p(1-p)]/∂θ = p(1-p)(1-2p)·[1, t̃]."""
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t_tilde)
    d_var = p * (1 - p) * (1 - 2 * p)
    return np.array([d_var, d_var * t_tilde])


# ═══════════════════════════════════════════════════════════════════
# PART 1: JACOBIAN VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_jacobians(n_test: int = 100) -> Dict[str, float]:
    """Validate closed-form Jacobians against autodiff."""
    from deep_inference.targets import DoseResponse, Profit, ConditionalVariance

    print("\n" + "=" * 60)
    print("PART 1: JACOBIAN VALIDATION (closed-form vs autodiff)")
    print("=" * 60)

    # Test points
    np.random.seed(42)
    thetas = np.random.randn(n_test, 2) * 0.5 + np.array([0.3, 1.0])
    t_tildes = [0.0, 0.5, 1.0, -0.5]

    targets = {
        "DoseResponse": (DoseResponse(model_type="logit"), oracle_jac_dose_response),
        "Profit": (Profit(model_type="logit"), oracle_jac_profit),
        "ConditionalVariance": (ConditionalVariance(model_type="logit"), oracle_jac_cond_var),
    }

    errors = {}
    for name, (target_obj, oracle_fn) in targets.items():
        max_err = 0.0
        for i in range(n_test):
            for t_val in t_tildes:
                theta_t = torch.tensor(thetas[i], dtype=torch.float32)
                x_t = torch.zeros(1)
                t_t = torch.tensor(t_val, dtype=torch.float32)

                # Closed-form
                jac_closed = target_obj.jacobian(x_t, theta_t, t_t)
                if jac_closed is None:
                    continue

                # Autodiff
                theta_ad = theta_t.detach().requires_grad_(True)
                h_val = target_obj.h(x_t, theta_ad, t_t)
                h_val.backward()
                jac_auto = theta_ad.grad.clone()

                err = torch.max(torch.abs(jac_closed - jac_auto)).item()
                max_err = max(max_err, err)

        errors[name] = max_err
        status = "PASS" if max_err < 1e-6 else "FAIL"
        print(f"  {name:<25} max_error = {max_err:.2e}  {status}")

    return errors


# ═══════════════════════════════════════════════════════════════════
# PART 2: COVERAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    sim_id: int
    target_name: str
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    covered: bool
    z_score: float


def run_single_sim(
    sim_id: int,
    n: int,
    target_name: str,
    t_tilde: float,
    mu_true: float,
    dgp: CanonicalDGP,
    n_folds: int,
    epochs: int,
    lambda_method: str,
) -> SimResult:
    """Run a single simulation for a specific target."""
    from deep_inference import inference

    Y, T, X, _, _ = generate_canonical_dgp(n=n, seed=sim_id, dgp=dgp)

    try:
        result = inference(
            Y=Y.numpy(),
            T=T.numpy(),
            X=X.numpy(),
            model="logit",
            target=target_name,
            t_tilde=t_tilde,
            lambda_method=lambda_method,
            n_folds=n_folds,
            epochs=epochs,
            patience=50,
            hidden_dims=[64, 32],
            lr=0.01,
            verbose=False,
        )

        mu_hat = result.mu_hat
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        covered = ci_lower <= mu_true <= ci_upper
        z_score = (mu_hat - mu_true) / se if se > 0 else np.nan

    except Exception as e:
        print(f"  Sim {sim_id} ({target_name}) FAILED: {e}")
        mu_hat = se = ci_lower = ci_upper = np.nan
        covered = False
        z_score = np.nan

    return SimResult(
        sim_id=sim_id,
        target_name=target_name,
        mu_hat=mu_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=covered,
        z_score=z_score,
    )


def compute_metrics(results: List[SimResult], mu_true: float) -> Dict:
    """Compute aggregate metrics from simulation results."""
    valid = [r for r in results if not np.isnan(r.mu_hat)]
    n_valid = len(valid)
    if n_valid == 0:
        return {"error": "All simulations failed"}

    mu_hats = np.array([r.mu_hat for r in valid])
    ses = np.array([r.se for r in valid])
    covered = np.array([r.covered for r in valid])
    z_scores = np.array([r.z_score for r in valid])
    z_scores = z_scores[~np.isnan(z_scores)]

    emp_se = mu_hats.std()
    mean_se = ses.mean()

    return {
        "n_simulations": len(results),
        "n_valid": n_valid,
        "n_failed": len(results) - n_valid,
        "mu_true": mu_true,
        "mean_mu_hat": mu_hats.mean(),
        "std_mu_hat": emp_se,
        "mean_se": mean_se,
        "emp_se": emp_se,
        "se_ratio": emp_se / mean_se if mean_se > 0 else np.nan,
        "bias": mu_hats.mean() - mu_true,
        "coverage": covered.mean(),
        "coverage_count": int(covered.sum()),
        "z_mean": z_scores.mean() if len(z_scores) > 0 else np.nan,
        "z_std": z_scores.std() if len(z_scores) > 0 else np.nan,
    }


def run_eval_10(
    M: int = 50,
    n: int = 8000,
    n_folds: int = 20,
    epochs: int = 200,
    lambda_method: str = "lgbm",
    n_jobs: int = 4,
    quick: bool = False,
):
    """
    Run new targets evaluation.

    Tests DoseResponse, Profit, ConditionalVariance with MC coverage.
    (TailProbability for logit is identical to DoseResponse, tested via Jacobian only.)
    """
    if quick:
        M = 10
        n = 3000
        epochs = 100

    print("=" * 70)
    print("EVAL 10: NEW TARGET FUNCTIONALS")
    print("=" * 70)

    dgp = CanonicalDGP()

    # ── Part 1: Jacobian validation ──
    jac_errors = validate_jacobians()
    jac_pass = all(e < 1e-6 for e in jac_errors.values())

    # ── Part 2: Coverage validation ──
    print("\n" + "=" * 60)
    print("PART 2: COVERAGE VALIDATION (Monte Carlo)")
    print("=" * 60)

    # Define targets with their oracles
    # NOTE: dose_response and conditional_variance at t̃=0 have degenerate
    # Jacobians (zero β-component). We test at t̃=0 with larger n to handle
    # the weaker IF correction, and also at t̃=0.5 where Jacobian is full-rank.
    target_configs = {
        "dose_response": {
            "t_tilde": 0.0,
            "mu_true": oracle_dose_response(dgp, 0.0),
        },
        "dose_response_t05": {
            "t_tilde": 0.5,
            "mu_true": oracle_dose_response(dgp, 0.5),
            "target_key": "dose_response",  # use same target string
        },
        "profit": {
            "t_tilde": 1.0,
            "mu_true": oracle_profit(dgp, 1.0),
        },
        "conditional_variance": {
            "t_tilde": 0.0,
            "mu_true": oracle_cond_var(dgp, 0.0),
        },
    }

    print(f"\nSettings: M={M}, n={n}, epochs={epochs}, lambda={lambda_method}")
    for name, cfg in target_configs.items():
        print(f"  {name}: t̃={cfg['t_tilde']}, μ*={cfg['mu_true']:.6f}")

    all_results = {}
    overall_pass = jac_pass

    for target_name, cfg in target_configs.items():
        # Support target_key override (e.g. "dose_response_t05" uses "dose_response")
        inference_target = cfg.get("target_key", target_name)
        print(f"\n{'─'*60}")
        print(f"TARGET: {target_name} (t̃={cfg['t_tilde']})")
        print(f"{'─'*60}")

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(run_single_sim)(
                sim_id=m,
                n=n,
                target_name=inference_target,
                t_tilde=cfg["t_tilde"],
                mu_true=cfg["mu_true"],
                dgp=dgp,
                n_folds=n_folds,
                epochs=epochs,
                lambda_method=lambda_method,
            )
            for m in range(1, M + 1)
        )

        metrics = compute_metrics(results, cfg["mu_true"])
        all_results[target_name] = metrics

        # Print metrics
        print(f"\n  μ*_true     = {metrics['mu_true']:.6f}")
        print(f"  Mean(μ̂)    = {metrics['mean_mu_hat']:.6f}")
        print(f"  Bias       = {metrics['bias']:.6f}")
        print(f"  Emp SE     = {metrics['emp_se']:.6f}")
        print(f"  Mean SE    = {metrics['mean_se']:.6f}")
        print(f"  SE Ratio   = {metrics['se_ratio']:.4f}")
        print(f"  Coverage   = {metrics['coverage']*100:.1f}% ({metrics['coverage_count']}/{metrics['n_valid']})")
        print(f"  z_mean     = {metrics['z_mean']:.4f}")
        print(f"  z_std      = {metrics['z_std']:.4f}")

        # Validate
        passed, criteria = validate_coverage_run(metrics)
        print(f"\n  Validation:")
        print(format_validation_table(criteria))
        if not passed:
            overall_pass = False

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("EVAL 10: SUMMARY")
    print("=" * 70)

    print(f"\n  Part 1 (Jacobians): {'PASS' if jac_pass else 'FAIL'}")
    for target_name, metrics in all_results.items():
        passed, _ = validate_coverage_run(metrics)
        status = "PASS" if passed else "FAIL"
        print(f"  {target_name}: Coverage={metrics['coverage']*100:.1f}%, SE_ratio={metrics['se_ratio']:.3f} → {status}")

    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    return {
        "jacobian_errors": jac_errors,
        "coverage_results": all_results,
        "passed": overall_pass,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--n", type=int, default=8000)
    args = parser.parse_args()

    result = run_eval_10(M=args.M, n=args.n, quick=args.quick)
