"""Probe: can the cholesky net fit the LOGIT Λ(x), in isolation from theta recovery?

Uses the TRUE per-obs logit Hessians (oracle theta) so any error is purely the net's
fit, not regularization bias in theta_hat. Compares Λ̂(x) (cholesky) and the heavy-ridge
baseline against the analytic oracle Λ*(x) on a held-out eval fold: Frobenius error,
eigenvalue spectrum (the low-overlap small eigenvalue is what the inverse is sensitive to),
and the inverse error that actually enters psi.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/probe_logit_lambda.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import (gen_logit, OracleLogitLambda, A0, A1, A2, B0, B1, GAMMA)  # noqa: E402
from deep_inference.lambda_.estimate import EstimateLambda  # noqa: E402


def true_per_obs_hessians(X, T):
    """H_i = w_i [[1,t],[t,t²]], w_i=p_i(1-p_i), p_i=σ(α*+β*·t)  (oracle theta)."""
    x0, x1 = X[:, 0], X[:, 1]
    a = A0 + A1 * x0 + A2 * x1
    b = B0 + B1 * x0
    p = torch.sigmoid(a + b * T)
    w = p * (1 - p)
    n = len(X)
    H = torch.zeros(n, 2, 2)
    H[:, 0, 0] = w
    H[:, 0, 1] = w * T
    H[:, 1, 0] = w * T
    H[:, 1, 1] = w * T * T
    return H


def fit_and_score(method, X_lam, H_lam, X_eval, Lam_star, **kw):
    est = EstimateLambda(method=method, **kw)
    est._d_theta = 2
    est._triu_idx = torch.triu_indices(2, 2)
    if method == "cholesky":
        est._fit_cholesky(X_lam, H_lam)
    elif method == "ridge":
        est._fit_ridge(X_lam, H_lam)
    Lam_hat = est.predict(X_eval)  # includes _project_to_psd
    # diagnostics
    frob = (Lam_hat - Lam_star).pow(2).sum().sqrt() / Lam_star.pow(2).sum().sqrt()
    inv_hat = torch.linalg.inv(Lam_hat + 1e-8 * torch.eye(2))
    inv_star = torch.linalg.inv(Lam_star + 1e-8 * torch.eye(2))
    inv_frob = (inv_hat - inv_star).pow(2).sum().sqrt() / inv_star.pow(2).sum().sqrt()
    eig_hat = torch.linalg.eigvalsh(Lam_hat)
    eig_star = torch.linalg.eigvalsh(Lam_star)
    return dict(frob=frob.item(), inv_frob=inv_frob.item(),
                min_eig_hat=eig_hat[:, 0].mean().item(),
                min_eig_star=eig_star[:, 0].mean().item(),
                max_invhat=inv_hat.abs().max().item(),
                max_invstar=inv_star.abs().max().item())


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    n = 6000
    rng = np.random.default_rng(0)
    Y, T, X = gen_logit(n, rng)
    Xt = torch.tensor(X, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    H = true_per_obs_hessians(Xt, Tt)

    half = n // 2
    X_lam, H_lam = Xt[:half], H[:half]
    X_eval = Xt[half:]
    Lam_star = OracleLogitLambda().predict(X_eval)

    print(f"oracle Λ*(x) on eval: mean min-eig={torch.linalg.eigvalsh(Lam_star)[:,0].mean():.4f} "
          f"max ||Λ*⁻¹||={torch.linalg.inv(Lam_star+1e-8*torch.eye(2)).abs().max():.1f}")
    print(f"{'method':14s} {'Λ-frob':>8s} {'Λinv-frob':>10s} {'minEig_hat':>11s} "
          f"{'minEig*':>9s} {'max|inv_hat|':>13s} {'max|inv*|':>10s}")
    for method, kw in [("cholesky", dict(chol_epochs=400, chol_lr=5e-3)),
                       ("cholesky", dict(chol_epochs=1000, chol_lr=5e-3)),
                       ("ridge", dict(ridge_alpha=1000.0))]:
        torch.manual_seed(1)
        d = fit_and_score(method, X_lam, H_lam, X_eval, Lam_star, **kw)
        tag = f"{method}" + (f"@{kw.get('chol_epochs')}" if method == "cholesky" else "")
        print(f"{tag:14s} {d['frob']:8.4f} {d['inv_frob']:10.4f} {d['min_eig_hat']:11.4f} "
              f"{d['min_eig_star']:9.4f} {d['max_invhat']:13.1f} {d['max_invstar']:10.1f}")


if __name__ == "__main__":
    main()
