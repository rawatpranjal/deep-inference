"""
Project the Lambda(x) shape out -- GENERAL, no e(x) shortcut.

Lambda(x) = E[l_theta_theta | X=x] is estimated the general way: regress the
per-observation Hessian ENTRIES on X (identity link, multi-output), reconstruct
the matrix, invert. No knowledge that the entries happen to be functions of one
scalar. We compare smooth regressors (GAM / spline / polynomial) against the
jagged (lgbm) and flat (ridge) ones, on the Hessian entries themselves.

Three panels, projected on the index s=X0+X1 (the only direction Lambda varies):
  (1) off-diagonal entry  Lambda_hat[0,1](s)   vs truth 2*sigmoid(s)   -- the shape
  (2) det Lambda_hat(s)                         vs truth 4*e(1-e)       -- CONDITIONING
      (this is what decides whether the inverse blows up; it must stay >0 and
       approach 0 smoothly in the low-overlap tails, not erratically)
  (3) 1/det Lambda_hat(s) (the variance weight) vs truth, log scale

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_surface.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from pygam import LinearGAM, s as gam_s

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import gen_linear, GAMMA


def fit_entries(kind, X, H):
    """Regress the Hessian upper-tri entries H=(n,3)=[h00,h01,h11] on X (general,
    identity link). Return predict: grid -> (m,3)."""
    if kind == "ridge@1000":
        r = Ridge(alpha=1000.0).fit(X, H); return r.predict
    if kind == "lgbm-heavy":
        r = MultiOutputRegressor(LGBMRegressor(n_estimators=20, max_depth=2, learning_rate=0.05,
            min_child_samples=150, reg_alpha=5.0, reg_lambda=5.0, verbose=-1)).fit(X, H)
        return r.predict
    if kind == "lgbm-light":
        r = MultiOutputRegressor(LGBMRegressor(n_estimators=300, num_leaves=63,
            min_child_samples=5, verbose=-1)).fit(X, H); return r.predict
    if kind == "poly-3":
        r = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1.0)).fit(X, H); return r.predict
    if kind == "spline":
        r = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=1.0)).fit(X, H)
        return r.predict
    if kind == "GAM":
        gams = [LinearGAM(gam_s(0) + gam_s(1) + gam_s(2) + gam_s(3) + gam_s(4)).fit(X, H[:, j])
                for j in range(H.shape[1])]
        return lambda G: np.column_stack([g.predict(G) for g in gams])
    raise ValueError(kind)


def main():
    rng = np.random.default_rng(3)
    Y, T, X = gen_linear(8000, rng)
    # general per-obs Hessian entries for the linear loss: [h00,h01,h11] = [2, 2T, 2T^2]
    H = np.column_stack([np.full(len(T), 2.0), 2 * T, 2 * T ** 2])

    s = np.linspace(-5, 5, 300)
    G = np.zeros((len(s), 5)); G[:, 0] = s / 2; G[:, 1] = s / 2
    e = 1.0 / (1.0 + np.exp(-GAMMA * s))
    off_true = 2 * e
    det_true = 2.0 * (2 * e) - (2 * e) ** 2          # h00*h11 - h01^2 = 4e - 4e^2 = 4e(1-e)

    estimators = ["oracle", "ridge@1000", "lgbm-heavy", "lgbm-light", "poly-3", "spline", "GAM"]
    colors = {"oracle": "white", "ridge@1000": "#e45756", "lgbm-heavy": "#f58518",
              "lgbm-light": "#ffbf79", "poly-3": "#54a24b", "spline": "#4c78a8", "GAM": "#b279a2"}

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    rows = []
    for kind in estimators:
        if kind == "oracle":
            off, det = off_true, det_true
        else:
            P = fit_entries(kind, X, H)(G)                 # (m,3)
            h00, h01, h11 = P[:, 0], P[:, 1], P[:, 2]
            off = h01
            det = h00 * h11 - h01 ** 2
        det_safe = np.where(det > 1e-6, det, 1e-6)
        lw = 2.6 if kind == "oracle" else 1.7
        ls = "--" if kind == "oracle" else "-"
        ax[0].plot(s, off, ls, color=colors[kind], lw=lw, label=kind)
        ax[1].plot(s, det, ls, color=colors[kind], lw=lw, label=kind)
        ax[2].plot(s, 1.0 / det_safe, ls, color=colors[kind], lw=lw, label=kind)
        if kind != "oracle":
            r2off = 1 - ((off - off_true) ** 2).sum() / ((off_true - off_true.mean()) ** 2).sum()
            neg = float((det <= 0).mean())
            rows.append((kind, r2off, neg, det.min()))

    ax[0].set_title("Lambda_hat[0,1](s) -- the off-diagonal shape"); ax[0].set_ylabel("Lambda[0,1]")
    ax[1].set_title("det Lambda_hat(s) -- conditioning (must stay > 0)"); ax[1].set_ylabel("det")
    ax[1].axhline(0, color="#888", lw=0.8, ls=":")
    ax[2].set_title("1/det -- the variance weight (log)"); ax[2].set_yscale("log"); ax[2].set_ylabel("1/det")
    for a in ax:
        a.set_xlabel("s = X0 + X1")
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("General Lambda(x): smooth bases fit the entry SHAPE (left) but independently-regressed "
                 "entries break PSD in the low-overlap tails -- det<0 (middle) -> 1/det spikes (right) -> "
                 "the inverse detonates. The constraint is conditioning, not smoothness.", fontsize=10)
    fig.tight_layout()
    out = "exploration/lambda_surface.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"wrote {out}")

    print("\nestimator      offdiag-R2   frac(det<=0)   min(det)   (det<0 -> non-PSD -> inverse detonates)")
    for kind, r2, neg, dmin in rows:
        print(f"  {kind:12s}   {r2:+.3f}        {neg:.2f}          {dmin:+.3f}")


if __name__ == "__main__":
    main()
