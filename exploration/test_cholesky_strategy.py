"""Self-check for the PSD-by-construction 'cholesky' lambda_method.

Two assertions on the new logic in EstimateLambda:
  (1) predict() is PSD (all eigenvalues > 0) for any X.
  (2) it recovers the conditional-mean Hessian E[H|X] from noisy per-obs Hessians
      (the net fits the mean, not the noisy targets).

Plus an end-to-end smoke that inference(model='logit', lambda_method='cholesky') and the
new lambda_strategy= passthrough both run and return finite SEs.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/test_cholesky_strategy.py
"""
import numpy as np
import torch

from deep_inference.lambda_.estimate import EstimateLambda
from deep_inference import inference


def _lam_star(X):
    """Known PSD conditional-mean Hessian Λ*(x)=[[1,e],[e,e+0.2]], e=σ(1.5·x0).
    det = (e+0.2) - e² > 0 for e∈(0,1), so PSD and well-conditioned."""
    e = torch.sigmoid(1.5 * X[:, 0])
    n, d = len(X), 2
    L = torch.zeros(n, d, d)
    L[:, 0, 0] = 1.0
    L[:, 0, 1] = e
    L[:, 1, 0] = e
    L[:, 1, 1] = e + 0.2
    return L


def check_strategy():
    torch.manual_seed(0)
    np.random.seed(0)
    n, d_x, d = 4000, 3, 2
    X = torch.randn(n, d_x)
    Lam = _lam_star(X)
    # per-obs Hessians = truth + symmetric mean-zero noise; the net must fit the MEAN
    noise = 0.3 * torch.randn(n, d, d)
    noise = 0.5 * (noise + noise.transpose(1, 2))
    H = Lam + noise

    est = EstimateLambda(method="cholesky", chol_epochs=400, chol_lr=5e-3)
    est._d_theta = d
    est._triu_idx = torch.triu_indices(d, d)
    est._fit_cholesky(X, H)

    Xte = torch.randn(2000, d_x)
    Lam_te = _lam_star(Xte)
    pred = est.predict(Xte)

    eig = torch.linalg.eigvalsh(pred)
    min_eig = eig.min().item()
    rel = (pred - Lam_te).pow(2).sum().sqrt() / Lam_te.pow(2).sum().sqrt()
    print(f"[strategy] min eigenvalue over 2000 preds: {min_eig:.4e}")
    print(f"[strategy] relative Frobenius error vs E[H|X]: {rel.item():.4f}")
    assert min_eig > 0, f"not PSD: min eig {min_eig}"
    assert rel.item() < 0.12, f"poor recovery: rel err {rel.item():.4f}"
    print("[strategy] PASS: PSD and recovers E[H|X]")


def check_inference_smoke():
    """Tiny logit run through the package: lambda_method='cholesky' and a
    lambda_strategy= passthrough both produce finite estimates/SEs."""
    rng = np.random.default_rng(1)
    n = 800
    X = rng.standard_normal((n, 3))
    T = (rng.uniform(size=n) < 0.5).astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 0.8 * X[:, 0] + (0.5 + 0.3 * X[:, 0]) * T)))
    Y = (rng.uniform(size=n) < p).astype(float)

    def ate(x, theta, t_tilde):
        return torch.sigmoid(theta[0] + theta[1]) - torch.sigmoid(theta[0])

    torch.manual_seed(0)
    r = inference(Y, T, X, model="logit", target_fn=ate, t_tilde=0.0,
                  lambda_method="cholesky", n_folds=3, epochs=40,
                  hidden_dims=[16], tikhonov_scale=1e-6, verbose=False)
    print(f"[inference] cholesky: mu={r.mu_hat:.4f} se={r.se:.4f}")
    assert np.isfinite(r.mu_hat) and np.isfinite(r.se) and r.se > 0

    # passthrough: hand inference() a pre-built strategy object directly
    strat = EstimateLambda(method="cholesky", chol_epochs=120)
    torch.manual_seed(0)
    r2 = inference(Y, T, X, model="logit", target_fn=ate, t_tilde=0.0,
                   lambda_strategy=strat, n_folds=3, epochs=40,
                   hidden_dims=[16], tikhonov_scale=1e-6, verbose=False)
    print(f"[inference] passthrough: mu={r2.mu_hat:.4f} se={r2.se:.4f}")
    assert np.isfinite(r2.mu_hat) and np.isfinite(r2.se) and r2.se > 0
    print("[inference] PASS: cholesky + passthrough both run")


if __name__ == "__main__":
    check_strategy()
    check_inference_smoke()
    print("\nALL CHECKS PASS")
