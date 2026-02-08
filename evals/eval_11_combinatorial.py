"""
Eval 11: Combinatorial Model (DeDL2025)

Validates CombinatorialModel link functions and MultiTreatmentATE target
using a DeDL-style DGP with m=3 binary treatments and gen_sigmoid_ii link.

DGP:
    X ~ Uniform(-2, 2)
    θ*(X) = [θ₀*(X), θ₁*(X), θ₂*(X), θ₃*(X), θ₄*(X)]
        θ₀ = 0.5 + 0.3·X   (intercept)
        θ₁ = 1.0 + 0.2·X   (treatment 1 effect)
        θ₂ = 0.5 - 0.1·X   (treatment 2 effect)
        θ₃ = 0.3 + 0.15·X  (treatment 3 effect)
        θ₄ = 5.0 + 0.5·X   (range parameter)
    T ~ each t_k independently Bernoulli(0.5)
    Y = G(θ*(X), T) + ε,  ε ~ N(0, 0.1²)

    G(θ, t) = θ₄ · σ(θ₀ + θ₁·t₁ + θ₂·t₂ + θ₃·t₃)   [gen_sigmoid_ii]

Part 1: Link function validation (G, loss, hessian)
Part 2: MultiTreatmentATE oracle validation
Part 3: Autodiff Jacobian consistency
"""

import sys
import numpy as np
import torch
from scipy.special import expit
from scipy import integrate
from typing import Dict, List

sys.path.insert(0, "/Users/pranjal/deepest/src")


# ═══════════════════════════════════════════════════════════════════
# DGP
# ═══════════════════════════════════════════════════════════════════

class CombinatorialDGP:
    """DGP for combinatorial experiments with gen_sigmoid_ii link."""

    def __init__(self):
        self.n_treatments = 3
        self.X_low = -2.0
        self.X_high = 2.0
        self.noise_std = 0.1

    def theta_star(self, x: np.ndarray) -> np.ndarray:
        """True θ*(X): [θ₀, θ₁, θ₂, θ₃, θ₄]."""
        n = len(x)
        theta = np.zeros((n, 5))
        theta[:, 0] = 0.5 + 0.3 * x    # intercept
        theta[:, 1] = 1.0 + 0.2 * x    # treatment 1
        theta[:, 2] = 0.5 - 0.1 * x    # treatment 2
        theta[:, 3] = 0.3 + 0.15 * x   # treatment 3
        theta[:, 4] = 5.0 + 0.5 * x    # range parameter
        return theta

    def G(self, theta: np.ndarray, t: np.ndarray) -> np.ndarray:
        """G(θ, t) = θ₄ · σ(θ₀ + θ₁·t₁ + θ₂·t₂ + θ₃·t₃)."""
        eta = theta[:, 0] + theta[:, 1] * t[:, 0] + theta[:, 2] * t[:, 1] + theta[:, 3] * t[:, 2]
        return theta[:, 4] * expit(eta)

    def G_single(self, theta_vec: np.ndarray, t_vec: np.ndarray) -> float:
        """G for a single observation."""
        eta = theta_vec[0] + theta_vec[1] * t_vec[0] + theta_vec[2] * t_vec[1] + theta_vec[3] * t_vec[2]
        return theta_vec[4] * expit(eta)

    def mu_true_ate(self, treatment: List[int], control: List[int] = None) -> float:
        """E[G(θ*(X), t) - G(θ*(X), t₀)] via numerical integration."""
        if control is None:
            control = [0, 0, 0]
        t = np.array(treatment, dtype=float)
        t0 = np.array(control, dtype=float)

        def integrand(x):
            theta = np.array([
                0.5 + 0.3 * x,
                1.0 + 0.2 * x,
                0.5 - 0.1 * x,
                0.3 + 0.15 * x,
                5.0 + 0.5 * x,
            ])
            g_t = theta[4] * expit(theta[0] + theta[1]*t[0] + theta[2]*t[1] + theta[3]*t[2])
            g_t0 = theta[4] * expit(theta[0] + theta[1]*t0[0] + theta[2]*t0[1] + theta[3]*t0[2])
            return (g_t - g_t0) * 0.25  # PDF of Uniform(-2, 2)

        result, _ = integrate.quad(integrand, self.X_low, self.X_high)
        return result

    def generate(self, n: int, seed: int = None):
        """Generate data from the DGP."""
        if seed is not None:
            np.random.seed(seed)

        X = np.random.uniform(self.X_low, self.X_high, n)
        theta_true = self.theta_star(X)

        # Treatment: each independently Bernoulli(0.5)
        T = np.random.binomial(1, 0.5, (n, 3)).astype(float)

        # Outcome
        Y = self.G(theta_true, T) + np.random.normal(0, self.noise_std, n)

        return Y, T, X, theta_true


# ═══════════════════════════════════════════════════════════════════
# PART 1: LINK FUNCTION VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_link_functions():
    """Validate all 4 link functions compute correctly."""
    from deep_inference.models import CombinatorialModel

    print("\n" + "=" * 60)
    print("PART 1: LINK FUNCTION VALIDATION")
    print("=" * 60)

    theta_test = torch.tensor([0.5, 1.0, 0.5, 0.3, 5.0], dtype=torch.float32)
    t_test = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    # gen_sigmoid_ii: θ₄ · σ(θ₀ + θ₁·t₁ + θ₂·t₂ + θ₃·t₃)
    model_ii = CombinatorialModel(n_treatments=3, link="gen_sigmoid_ii")
    g_ii = model_ii.G(theta_test, t_test)
    expected_ii = 5.0 * expit(0.5 + 1.0*1.0 + 0.5*0.0 + 0.3*1.0)
    err_ii = abs(g_ii.item() - expected_ii)
    print(f"  gen_sigmoid_ii: G={g_ii.item():.6f}, expected={expected_ii:.6f}, err={err_ii:.2e}  {'PASS' if err_ii < 1e-5 else 'FAIL'}")

    # multiplicative: θ₀ · ∏(1 + θ_k · t_k)
    theta_mult = torch.tensor([2.0, 0.5, 0.3, 0.1], dtype=torch.float32)
    model_mult = CombinatorialModel(n_treatments=3, link="multiplicative")
    g_mult = model_mult.G(theta_mult, t_test)
    expected_mult = 2.0 * (1 + 0.5*1.0) * (1 + 0.3*0.0) * (1 + 0.1*1.0)
    err_mult = abs(g_mult.item() - expected_mult)
    print(f"  multiplicative: G={g_mult.item():.6f}, expected={expected_mult:.6f}, err={err_mult:.2e}  {'PASS' if err_mult < 1e-5 else 'FAIL'}")

    # sigmoid: a/(1+exp(-(θ₀ + Σθ_k·t_k))) + b
    theta_sig = torch.tensor([0.5, 1.0, 0.5, 0.3], dtype=torch.float32)
    model_sig = CombinatorialModel(n_treatments=3, link="sigmoid", a=2.0, b=1.0)
    g_sig = model_sig.G(theta_sig, t_test)
    expected_sig = 2.0 * expit(0.5 + 1.0*1.0 + 0.5*0.0 + 0.3*1.0) + 1.0
    err_sig = abs(g_sig.item() - expected_sig)
    print(f"  sigmoid:        G={g_sig.item():.6f}, expected={expected_sig:.6f}, err={err_sig:.2e}  {'PASS' if err_sig < 1e-5 else 'FAIL'}")

    # gen_sigmoid_i: θ_{m+1} · σ(Σθ_k·t_k)  [no intercept]
    # theta = [θ₁, θ₂, θ₃, θ₄] (0-indexed: theta[0..2] = coefficients, theta[3] = range)
    theta_gs1 = torch.tensor([1.0, 0.5, 0.3, 5.0], dtype=torch.float32)
    model_gs1 = CombinatorialModel(n_treatments=3, link="gen_sigmoid_i")
    g_gs1 = model_gs1.G(theta_gs1, t_test)
    expected_gs1 = 5.0 * expit(1.0*1.0 + 0.5*0.0 + 0.3*1.0)
    err_gs1 = abs(g_gs1.item() - expected_gs1)
    print(f"  gen_sigmoid_i:  G={g_gs1.item():.6f}, expected={expected_gs1:.6f}, err={err_gs1:.2e}  {'PASS' if err_gs1 < 1e-5 else 'FAIL'}")

    # Validate loss = (y - G)²
    y_test = torch.tensor(3.0, dtype=torch.float32)
    loss_ii = model_ii.loss(y_test, t_test, theta_test)
    expected_loss = (3.0 - expected_ii) ** 2
    err_loss = abs(loss_ii.item() - expected_loss)
    print(f"\n  Loss (gen_sigmoid_ii): {loss_ii.item():.6f}, expected={expected_loss:.6f}, err={err_loss:.2e}  {'PASS' if err_loss < 1e-5 else 'FAIL'}")

    # Validate Hessian = 2·G_θ·G_θ' (Fisher information)
    hess = model_ii.hessian(y_test, t_test, theta_test)
    # Compute G_θ via autodiff separately
    theta_req = theta_test.detach().requires_grad_(True)
    g_val = model_ii.G(theta_req, t_test)
    g_theta = torch.autograd.grad(g_val, theta_req)[0]
    expected_hess = 2 * torch.outer(g_theta, g_theta)
    err_hess = torch.max(torch.abs(hess - expected_hess)).item()
    print(f"  Hessian (Fisher): max_err={err_hess:.2e}  {'PASS' if err_hess < 1e-5 else 'FAIL'}")

    # Hessian should NOT depend on y
    hess_other_y = model_ii.hessian(torch.tensor(100.0), t_test, theta_test)
    err_y_indep = torch.max(torch.abs(hess - hess_other_y)).item()
    print(f"  Hessian y-invariance: err={err_y_indep:.2e}  {'PASS' if err_y_indep < 1e-10 else 'FAIL'}")

    all_pass = all(e < 1e-5 for e in [err_ii, err_mult, err_sig, err_gs1, err_loss, err_hess]) and err_y_indep < 1e-10
    return all_pass


# ═══════════════════════════════════════════════════════════════════
# PART 2: ORACLE ATE VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_oracle_ates():
    """Validate MultiTreatmentATE target against oracle via MC."""
    from deep_inference.models import CombinatorialModel
    from deep_inference.targets import MultiTreatmentATE

    print("\n" + "=" * 60)
    print("PART 2: ORACLE ATE VALIDATION")
    print("=" * 60)

    dgp = CombinatorialDGP()
    model = CombinatorialModel(n_treatments=3, link="gen_sigmoid_ii")

    # Test treatment combinations
    treatments = [
        [1, 0, 0],  # only treatment 1
        [0, 1, 0],  # only treatment 2
        [0, 0, 1],  # only treatment 3
        [1, 1, 0],  # treatments 1 + 2
        [1, 1, 1],  # all treatments
    ]

    print(f"\n  {'Treatment':<15} {'Oracle μ*':<12} {'MC μ̂':<12} {'Error':<12} {'Status'}")
    print("  " + "-" * 60)

    all_pass = True
    np.random.seed(42)
    n_mc = 100000
    X_mc = np.random.uniform(dgp.X_low, dgp.X_high, n_mc)
    theta_mc = dgp.theta_star(X_mc)

    for t_vec in treatments:
        # Oracle (numerical integration)
        mu_oracle = dgp.mu_true_ate(t_vec)

        # MC estimate
        t_arr = np.tile(t_vec, (n_mc, 1))
        t0_arr = np.zeros((n_mc, 3))
        g_t = dgp.G(theta_mc, t_arr)
        g_t0 = dgp.G(theta_mc, t0_arr)
        mu_mc = (g_t - g_t0).mean()

        # Target object validation
        target = MultiTreatmentATE(model=model, treatment=t_vec)
        theta_t = torch.tensor(theta_mc[0], dtype=torch.float32)
        h_val = target.h(torch.zeros(1), theta_t, torch.tensor(0.0))

        # Oracle single-point check
        g_t_single = dgp.G_single(theta_mc[0], np.array(t_vec, dtype=float))
        g_t0_single = dgp.G_single(theta_mc[0], np.array([0, 0, 0], dtype=float))
        expected_h = g_t_single - g_t0_single

        err = abs(mu_oracle - mu_mc)
        err_h = abs(h_val.item() - expected_h)

        status = "PASS" if err < 0.01 and err_h < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False

        t_str = str(t_vec)
        print(f"  {t_str:<15} {mu_oracle:<12.6f} {mu_mc:<12.6f} {err:<12.6f} {status}")

    return all_pass


# ═══════════════════════════════════════════════════════════════════
# PART 3: AUTODIFF JACOBIAN CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

def validate_ate_jacobians():
    """Validate MultiTreatmentATE Jacobian via autodiff."""
    from deep_inference.models import CombinatorialModel
    from deep_inference.targets import MultiTreatmentATE

    print("\n" + "=" * 60)
    print("PART 3: AUTODIFF JACOBIAN CONSISTENCY")
    print("=" * 60)

    model = CombinatorialModel(n_treatments=3, link="gen_sigmoid_ii")

    treatments = [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]

    np.random.seed(42)
    n_test = 50
    thetas = np.random.randn(n_test, 5) * 0.3 + np.array([0.5, 1.0, 0.5, 0.3, 5.0])

    all_pass = True
    for t_vec in treatments:
        target = MultiTreatmentATE(model=model, treatment=t_vec)
        max_rel_err = 0.0

        for i in range(n_test):
            theta_t = torch.tensor(thetas[i], dtype=torch.float64, requires_grad=True)
            x_t = torch.zeros(1, dtype=torch.float64)
            t_t = torch.tensor(0.0, dtype=torch.float64)

            # Ensure model tensors match dtype
            target_64 = MultiTreatmentATE(model=model, treatment=t_vec)

            # Forward
            h_val = target_64.h(x_t, theta_t, t_t)
            h_val.backward()
            jac_auto = theta_t.grad.clone()

            # Numerical Jacobian (central differences, float64)
            eps = 1e-7
            jac_num = torch.zeros(5, dtype=torch.float64)
            for j in range(5):
                theta_plus = thetas[i].copy()
                theta_plus[j] += eps
                theta_minus = thetas[i].copy()
                theta_minus[j] -= eps

                tp = torch.tensor(theta_plus, dtype=torch.float64)
                tm = torch.tensor(theta_minus, dtype=torch.float64)
                jac_num[j] = (target_64.h(x_t, tp, t_t) - target_64.h(x_t, tm, t_t)) / (2 * eps)

            # Relative error (avoid division by zero)
            scale = torch.max(torch.abs(jac_auto), torch.abs(jac_num)).clamp(min=1e-10)
            rel_err = torch.max(torch.abs(jac_auto - jac_num) / scale).item()
            max_rel_err = max(max_rel_err, rel_err)

        status = "PASS" if max_rel_err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Treatment {t_vec}: max_rel_error = {max_rel_err:.2e}  {status}")

    return all_pass


# ═══════════════════════════════════════════════════════════════════
# PART 4: COMPUTE_LAMBDA_INTEGRAL VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_lambda_integral():
    """Validate compute_lambda_integral for Regime A."""
    from deep_inference.models import CombinatorialModel

    print("\n" + "=" * 60)
    print("PART 4: COMPUTE_LAMBDA_INTEGRAL (Regime A)")
    print("=" * 60)

    model = CombinatorialModel(n_treatments=3, link="gen_sigmoid_ii")
    theta_test = torch.tensor([0.5, 1.0, 0.5, 0.3, 5.0], dtype=torch.float32)

    # Generate all 8 treatment combinations (2³)
    all_combos = torch.tensor([
        [i, j, k] for i in range(2) for j in range(2) for k in range(2)
    ], dtype=torch.float32)

    # compute_lambda_integral should average over these
    Lambda = model.compute_lambda_integral(theta_test, all_combos)

    # Manual computation
    Lambda_manual = torch.zeros(5, 5)
    for i in range(8):
        t = all_combos[i]
        theta_req = theta_test.detach().requires_grad_(True)
        g = model.G(theta_req, t)
        g_theta = torch.autograd.grad(g, theta_req)[0]
        Lambda_manual += 2 * torch.outer(g_theta, g_theta)
    Lambda_manual /= 8

    err = torch.max(torch.abs(Lambda - Lambda_manual)).item()
    print(f"  Lambda integral error: {err:.2e}  {'PASS' if err < 1e-6 else 'FAIL'}")

    # Check PSD
    eigvals = torch.linalg.eigvalsh(Lambda)
    min_eig = eigvals.min().item()
    print(f"  Min eigenvalue: {min_eig:.6f}  {'PASS' if min_eig > 0 else 'FAIL'}")

    # Check shape
    print(f"  Shape: {Lambda.shape} (expected: (5, 5))  {'PASS' if Lambda.shape == (5, 5) else 'FAIL'}")

    return err < 1e-6 and min_eig > 0


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_eval_11(quick: bool = False):
    """Run all combinatorial model validations."""
    print("=" * 70)
    print("EVAL 11: COMBINATORIAL MODEL (DeDL2025)")
    print("=" * 70)

    p1 = validate_link_functions()
    p2 = validate_oracle_ates()
    p3 = validate_ate_jacobians()
    p4 = validate_lambda_integral()

    overall = p1 and p2 and p3 and p4

    print("\n" + "=" * 70)
    print("EVAL 11: SUMMARY")
    print("=" * 70)
    print(f"  Part 1 (Link functions):    {'PASS' if p1 else 'FAIL'}")
    print(f"  Part 2 (Oracle ATEs):       {'PASS' if p2 else 'FAIL'}")
    print(f"  Part 3 (ATE Jacobians):     {'PASS' if p3 else 'FAIL'}")
    print(f"  Part 4 (Lambda integral):   {'PASS' if p4 else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    return {"passed": overall}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    result = run_eval_11(quick=args.quick)
