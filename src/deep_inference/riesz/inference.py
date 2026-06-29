"""Cross-fit RieszNet inference for the average treatment effect.

`riesz_inference` estimates the binary-treatment ATE contrast

    mu = E[ g(1, X) - g(0, X) ]

with the doubly-robust moment of Chernozhukov et al. (2022, Section 5),

    psi_i = (g_tilde(1, x_i) - g_tilde(0, x_i)) + a(z_i) (y_i - g_tilde(z_i)),

averaged over a K-fold cross-fit and combined across repeated splits with the median-DML
rule. The representer a and the targeted-regularized regression g_tilde come from the
RieszNet model in `model.py`. "Doubly robust" means the estimate stays valid if either g
or a is learned well, not necessarily both.

This is a faithful port of the validated prototype in exploration/spike.py.
"""

from __future__ import annotations

import numpy as np
import torch

from ..engine.variance import compute_se_ci, median_dml_aggregate
from .model import L2, RieszNet, _torch_Phi, combined_loss


def _fit_riesz(
    Xtr,
    Ttr,
    Ytr,
    outcome,
    max_epochs,
    patience,
    seed,
    nb_dispersion=3.0,
    restarts=3,
    batch_size=256,
    weight_decay=L2,
):
    """Fit one RieszNet on a training fold.

    Minibatch Adam with a two-stage learning rate (1e-3 then 2e-4) and early stopping on
    a held-out 20% split. Restart selection uses a divergence-robust validation metric:
    the combined validation loss plus a soft penalty on a runaway representer. The DR
    moment blows up when the learned representer a(Z) runs away, while the true ATE
    representer (T - e)/(e(1 - e)) is bounded by overlap, so an implausibly large
    validation |a| flags a junk head that the plain validation loss would not reject.
    This guard is truth-free.
    """
    d_x = Xtr.shape[1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ytr))
    n_val = max(1, int(0.2 * len(Ytr)))
    val, tr = idx[:n_val], idx[n_val:]
    n_tr = len(tr)
    stage2 = max_epochs // 2  # learning-rate drop point

    def pack(ii):
        x = torch.tensor(Xtr[ii], dtype=torch.float32)
        t = torch.tensor(Ttr[ii], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(Ytr[ii], dtype=torch.float32)
        return t, x, y, torch.ones_like(t), torch.zeros_like(t)

    val_b = pack(val)

    def val_score(net):
        with torch.no_grad():
            loss = combined_loss(net, *val_b, outcome, nb_dispersion).item()
            _, a_val = net(val_b[0], val_b[1])
            a_rms = float((a_val**2).mean().sqrt())
        return loss + 0.01 * max(0.0, a_rms - 25.0) ** 2

    global_best, global_state = float("inf"), None
    for r in range(restarts):
        torch.manual_seed(seed + 1000 * r)
        brng = np.random.default_rng(seed + 7 * r + 1)
        net = RieszNet(d_x)
        # L2 on net weights only, NOT on eps (the paper does not penalize eps).
        opt = torch.optim.Adam(
            [
                {
                    "params": [p for n, p in net.named_parameters() if n != "eps"],
                    "weight_decay": weight_decay,
                },
                {"params": [net.eps], "weight_decay": 0.0},
            ],
            lr=1e-3,
        )
        best, best_state, wait = float("inf"), None, 0
        for ep in range(max_epochs):
            for g in opt.param_groups:
                g["lr"] = 1e-3 if ep < stage2 else 2e-4
            net.train()
            perm = tr[brng.permutation(n_tr)]
            for s in range(0, n_tr, batch_size):
                bb = pack(perm[s : s + batch_size])
                opt.zero_grad()
                combined_loss(net, *bb, outcome, nb_dispersion).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            v = val_score(net)
            if v < best - 1e-5:
                best, best_state, wait = (
                    v,
                    {k: p.clone() for k, p in net.state_dict().items()},
                    0,
                )
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None and best < global_best:
            global_best, global_state = best, best_state
    net = RieszNet(d_x)
    if global_state is not None:
        net.load_state_dict(global_state)
    net.eval()
    return net


def _riesz_single(Y, T, X, outcome, K, max_epochs, patience, seed, nb_dispersion=3.0):
    """One K-fold cross-fit pass. Returns (mu, se, mu_naive, se_naive, psi, a_obs, g_diff)."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), K)
    psi = np.zeros(n)
    g_diff = np.zeros(n)
    a_full = np.zeros(n)
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        net = _fit_riesz(
            X[tr], T[tr], Y[tr], outcome, max_epochs, patience, seed + k, nb_dispersion
        )
        with torch.no_grad():
            xe = torch.tensor(X[te], dtype=torch.float32)
            te_t = torch.tensor(T[te], dtype=torch.float32).unsqueeze(1)
            ones, zeros = torch.ones_like(te_t), torch.zeros_like(te_t)
            eps = float(np.clip(net.eps.item(), -2.0, 2.0))

            def gmean(traw):
                g, a = net(traw, xe)
                if outcome == "logit":
                    g = torch.sigmoid(g)
                elif outcome in ("poisson", "gamma", "negbin"):
                    g = torch.exp(g)
                elif outcome == "probit":
                    g = _torch_Phi(g)
                return g.numpy(), a.numpy()

            g1, a1 = gmean(ones)
            g0, a0 = gmean(zeros)
            g_obs, a_obs = gmean(te_t)
        # Targeted-regularized regression, then the DR moment. Poisson fluctuates on the
        # log scale (g_tilde = g * exp(eps*a)); other outcomes on the mean scale.
        if outcome == "poisson":
            gt1 = g1 * np.exp(eps * a1)
            gt0 = g0 * np.exp(eps * a0)
            gt_obs = g_obs * np.exp(eps * a_obs)
        else:
            gt1 = g1 + eps * a1
            gt0 = g0 + eps * a0
            gt_obs = g_obs + eps * a_obs
        g_diff[te] = g1 - g0  # uncorrected plug-in contrast
        psi[te] = (gt1 - gt0) + a_obs * (Y[te] - gt_obs)  # DR moment
        a_full[te] = a_obs
    se, _lo, _hi, _ = compute_se_ci(torch.tensor(psi))
    mu = float(psi.mean())
    mu_naive = float(g_diff.mean())
    se_naive = float(g_diff.std(ddof=1) / np.sqrt(n))
    return mu, float(se), mu_naive, se_naive, psi, a_full, g_diff


def riesz_inference(
    Y,
    T,
    X,
    outcome: str = "linear",
    n_folds: int = 5,
    n_repeats: int = 3,
    max_epochs: int = 400,
    patience: int = 30,
    seed: int = 0,
    nb_dispersion: float = 3.0,
    alpha: float = 0.05,
    store_data: bool = True,
    verbose: bool = False,
):
    """Automatic-debiasing inference for the ATE via RieszNet.

    Args:
        Y: (n,) outcomes.
        T: (n,) treatment. The estimand is the contrast E[g(1, X) - g(0, X)], so for a
            binary treatment this is the ATE; for a continuous treatment it is the effect
            of moving T from 0 to 1.
        X: (n, d) covariates.
        outcome: one of "linear", "logit", "poisson", "gamma", "negbin", "probit".
        n_folds: cross-fitting folds K.
        n_repeats: independent split repeats, combined with the median-DML rule.
        max_epochs, patience: training schedule per fold.
        seed: base random seed.
        nb_dispersion: the negative-binomial size r (only used when outcome="negbin").
        alpha: CI level (0.05 gives a 95% interval).
        store_data: keep X and the learned representer on the result for inspection.
        verbose: print per-repeat progress.

    Returns:
        InferenceResult with mu_hat, se, ci_lower, ci_upper, psi_values (the DR moment
        from the first repeat), theta_hat (the learned representer a(Z), first repeat),
        and a diagnostics dict.
    """
    Y = np.asarray(Y, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(Y)

    results: list[tuple] = []
    for r in range(n_repeats):
        out = _riesz_single(
            Y,
            T,
            X,
            outcome,
            n_folds,
            max_epochs,
            patience,
            seed + 100 * r,
            nb_dispersion,
        )
        results.append(out)
        if verbose:
            print(
                f"[riesz] repeat {r + 1}/{n_repeats}: mu={out[0]:.4f} se={out[1]:.4f}"
            )

    mus = [r[0] for r in results]
    ses = [r[1] for r in results]
    mu_hat, se, ci_lower, ci_upper = median_dml_aggregate(mus, ses, alpha=alpha)

    # Per-observation arrays from the first repeat (representative; the headline stats
    # come from the median-DML aggregate across all repeats).
    _, _, mu_naive, se_naive, psi, a_full, g_diff = results[0]

    diagnostics = {
        "procedure": "riesznet",
        "outcome": outcome,
        "regime": "DR (automatic debiasing)",
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "mu_naive": mu_naive,
        "se_naive": se_naive,
        "mean_abs_representer": float(np.mean(np.abs(a_full))),
        "repeat_estimates": [float(m) for m in mus],
    }

    from .. import InferenceResult

    return InferenceResult(
        mu_hat=float(mu_hat),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        psi_values=torch.tensor(psi, dtype=torch.float32),
        theta_hat=torch.tensor(a_full, dtype=torch.float32).reshape(-1, 1),
        diagnostics=diagnostics,
        _model=f"riesznet[{outcome}]",
        _target="ATE",
        _n_obs=n,
        _n_folds=n_folds,
        _X_train=X if store_data else None,
        _theta_predictor=None,
    )
