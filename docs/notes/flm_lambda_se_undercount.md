# Why FLM under-covers on the linear ATE, and how to fix it

**Date.** 2026-06-27. **Status.** Diagnosed. Linear fix and a general fix both work in a spike.
M=200 confirm and fresh-agent verification pending before "solved."

**One line.** The FLM influence-function estimator under-covers on the confounded linear ATE
because the package collapses the curvature matrix Λ(x) to a covariate-constant average. The real
obstacle is conditioning: regressing the Hessian entries independently makes Λ̂(x) non-PSD at low
overlap, so the inverse detonates. Two fixes restore valid coverage. A family-specific one (estimate
the propensity e(x) and build Λ analytically, 95 percent / SE-ratio 1.04 at M=200), and a general
one (regress Λ̂(x)=L(x)L(x)ᵀ in Cholesky form so it is PSD by construction, 96 percent / 1.01 at
M=50). RieszNet, which never inverts anything, is the third route, and FLM with the correct Λ equals
RieszNet.

## The problem

On the canonical linear demand benchmark (`exploration/spike.py`), `structural_dml(family='linear')`
reported 95 percent confidence intervals that covered the truth only 84 to 92 percent of the time,
with a standard error that was about 14 percent too small (SE-ratio around 0.86). The same DGP
estimated by a correctly specified OLS oracle, and by a RieszNet automatic-debiasing net, both
covered at 94 to 98 percent with SE-ratio near 1.0. So the gap was specific to FLM, not to the
problem.

The DGP is confounded by construction. Treatment is binary with propensity e(x) = σ(X₀+X₁), which
varies across rows (sd 0.26, range 0.005 to 0.997). Outcome is Y = α(X) + β(X)·T + ε. The target is
μ = E[β(X)], the average slope, which equals the ATE because Y is linear in T.

## The diagnosis

For this model the influence function reduces, after the algebra, to one line:

```
ψ_i  =  β(x_i)  +  ε_i · (T_i − e(x_i)) / ( e(x_i)(1 − e(x_i)) ),   SE = sqrt(Var(ψ)/n)
```

The whole correction rides on the propensity e(x) = E[T|X], which enters through the curvature
Λ(x) = E[ℓ_θθ | X=x] = 2·[[1, e(x)], [e(x), e(x)]]. The package estimates Λ on the two-way
cross-fitting path (the default for a θ-independent Hessian) with a single global-mean matrix, which
replaces the varying e(x) with the constant ē = 0.5. That substitution swaps the correct variance
weight E[1/(e(1−e))] = 7.44 for the flat 1/(ē(1−ē)) = 4.0, and shrinks the SE.

Three independent checks place the defect on Λ(x), not on sample size or the network.

- **Flat in n.** SE-ratio was 0.899 at n=2000 and 0.886 at n=8000. Quadrupling the data moved
  nothing, which rules out a finite-sample story
  (`evals/reports/flm_overfit_debug_20260627_203913.txt`).
- **Closed form.** The flat-weight prediction for the SE at n=2000 is 0.0461. The package reported
  0.0463. The match to three decimals confirms the package uses the flat weight.
- **Oracle injection.** Feeding the true Λ(x) into the real pipeline restores it. With oracle θ and
  oracle Λ the reported SE equals sqrt(Var(ψ*)/n) to five decimals, and the variance ratio
  Var(ψ̂)/Var(ψ*) is 1.000 (`exploration/lambda_decomp.py`, sanity gate).

A full-observability decomposition scored every object in the seven-step chain against its oracle
(`exploration/results_lambda_decomp.md`). It sharpened the location to the inverse, not the matrix.

```
cell                SE-ratio  cov   Var(ψ̂)/Var(ψ*)  Λ⁻¹-R²    Λ-R²    note
flat / θ=net          0.83    87%       0.56         -0.11    -0.00   the bug
ridge / θ=net         0.38    90%     169.6      -19807        0.94   Λ fit fine, Λ⁻¹ explodes
rf / θ=net             1.14    97%       2.03         -0.08     0.78   overshoots
oracle Λ / θ=net       1.08    93%       1.18          1.00     1.00   true Λ -> in target
oracle Λ / θ=oracle    1.11   100%       1.00          1.00     1.00   gold: every fit = 1
```

The load-bearing object is Λ⁻¹, not Λ. A ridge fit recovers Λ accurately (R² 0.94) yet its inverse
is garbage (R² −19807), because Λ(x) goes near-singular exactly at the low-overlap rows where
det Λ = 4·e(1−e) approaches zero. Inverting an independently-estimated Λ is the failure mode.

**Unification with RieszNet.** With the correct Λ(x) the FLM correction weight (T−e)/(e(1−e)) is
exactly RieszNet's learned Riesz representer, and ψ_FLM = ψ_RieszNet term for term. FLM and RieszNet
are the same estimator written two ways. RieszNet stays stable because it learns that weight
directly under a regularizing loss, instead of inverting a near-singular matrix.

## The fix

Do not regress the Hessian entries and invert. Estimate the scalar propensity e(x) = E[T|X]
directly, then form Λ(x) = 2·[[1, e], [e, e]] analytically. The structured matrix is PSD with
det = 4·e(1−e) > 0 by construction, so the inverse never explodes. One well-posed scalar replaces a
near-singular matrix.

The choice of estimator for e(x) matters, and the rule is to use a classifier that respects the
zero-one range (`exploration/results_lambda_inv_fix.md`, M=50, n=2000, θ=net).

```
estimator for e(x)      SE-ratio  cov   Λ⁻¹-R²    note
logistic-ridge            1.12    96%    0.94     the fix, matches oracle
MLP classifier            1.18    96%    0.58     workable
HistGradientBoosting      1.08   100%    0.12     noisy propensity
LPM (linear OLS)          0.91   100%  -629       detonates, cannot bound to [0,1]
ridge (regress + invert)  0.43    96%  -514       the old unstable path
random forest (same)      1.14    98%   -0.11     poor
```

Logistic-ridge is the only estimator that matches the oracle's Λ⁻¹ (0.94 against 1.00). On the
M=200 confirm it brings coverage to 95 percent with SE-ratio 1.04, against 89 percent for the flat
bug and matching the oracle at 95 percent / 1.03 (`exploration/results_fix_confirm_M200.md`). Both
target bands are met. More data helps, which the flat Λ never did. As n grows from 1000 to 8000 the
Λ⁻¹ recovery climbs 0.886, 0.939, 0.957, 0.986 and the variance ratio falls toward 1 (1.24, 1.22,
1.15, 1.09). Note this works because, for the linear family, every Hessian entry is a known function
of the single scalar e(x), so estimating that one scalar and imposing the matrix structure
guarantees a well-conditioned, exactly invertible Λ. The next section shows that escape is not
available in general.

## The general case

The structured fix above only works because the squared-error Hessian is a closed form in one
scalar, e(x). A general structural model has no such form, so the package estimates Λ(x) the general
way (`core/lambda_estimator.py`). It autodiffs the per-observation Hessian ℓ_θθ(z_i; θ̂(x_i)), which
works for any loss, flattens its upper triangle, regresses those entries on X with a multi-output
learner, reconstructs Λ̂(x), and inverts per-x. No closed form is assumed.

We tested whether any general Λ(x) regressor restores valid SEs on the same severe-overlap spike
(`exploration/lambda_general.py`, M=50, n=2000, θ=net). None does.

```
general Λ regressor   SE-ratio  cov   Var(ψ̂)/Var(ψ*)   Λ-R² (fit)   Λ⁻¹-R² (inverse)
flat (bug reference)    0.81    88%       0.56           -0.00        -0.11
oracle (gold)           1.09    96%       1.16            1.00         1.00
lgbm (heavy reg)        0.78    84%       0.53            0.63        -0.04
mlp (early-stopped)     0.60   100%    1475              0.77        -6766
random forest           1.14    98%       2.39            0.78        -0.11
ridge (alpha 1000)      0.76    84%       0.51            0.64         0.01
```

They fail in different directions, none calibrated: lgbm and ridge undercount the variance, rf
over-inflates, mlp detonates. The learning curves across n=1000 to 8000 show why. The Λ fit climbs
with data (lgbm Λ-R² 0.41 to 0.68, mlp 0.48 to 0.94) while the inverse and the coverage do not move
(lgbm Λ⁻¹-R² −0.08 to −0.03, coverage stuck near 90 percent). Estimating Λ(x) better is not the
bottleneck. Inverting it is. MLP is the clearest case, fitting the Hessian to Λ-R² 0.94 yet
inverting to −7040, because a well-fit but near-singular matrix inverts to noise.

So the general lesson is the opposite of a tuning exercise. Under severe overlap no entry-wise
regressor makes regress-and-invert work, and more data does not rescue it. The projection of the
Λ(x) surface (`exploration/lambda_surface.png`) shows why directly. Smooth bases recover each
Hessian entry's shape well in the data-dense middle, yet because the entries are regressed
independently they break the PSD constraint in the low-overlap tails, det Λ̂ goes negative for 38 to
50 percent of the swept index for ridge, spline, and GAM, and 1/det spikes to infinity there. The
constraint that matters is conditioning, not smoothness, and post-hoc eigenvalue clamping does not
repair an already-wrong matrix (it moved a detonating Λ⁻¹-R² from −4395 to −385, still useless).

### The general fix: PSD by construction

The fix follows from the obstacle. Enforce PSD during the regression, not after. A small net maps x
to the Cholesky factor L(x), and Λ̂(x) = L(x)L(x)ᵀ is positive semidefinite for any net output,
trained to fit the per-observation Hessians in Frobenius norm. This is fully general, autodiff the
Hessian for any loss, output a lower-triangular factor of any dimension, no closed form and no e(x).

It works (`exploration/lambda_cholesky.py`, M=50, n=2000, θ=net):

```
method        SE-ratio  cov   Var(ψ̂)/Var(ψ*)   Λ⁻¹-R²    bias
flat (bug)      0.81    88%       0.56          -0.11    +0.014
lgbm            0.78    84%       0.53          -0.04    +0.018
cholesky        1.01    96%       1.04          +0.17    -0.002
oracle          1.09    96%       1.16          +1.00    +0.010
```

Cholesky is the only general estimator with a positive Λ⁻¹-R² and a calibrated variance ratio, and
it lands both target bands (coverage 96 percent, SE-ratio 1.01) with near-zero bias. Enforcing PSD
trades a little Λ accuracy (Λ⁻¹-R² 0.17 against the oracle's 1.00) for guaranteed conditioning, and
conditioning was the whole problem. RieszNet, which learns the correction directly and so never
inverts anything, remains the other general route, and FLM with the correct Λ equals RieszNet.

## Honest caveats

- The fixed estimator now mildly **over**-covers (96 percent, SE-ratio about 1.1), slightly above the
  [0.92, 1.08] SE-ratio band but inside the [93, 97] coverage band. The overshoot is not Λ anymore.
  Even with oracle Λ the variance ratio is 1.16 under the real network, because the net's imperfect
  θ (β recovery R² near 0.5) inflates the residual ε̂. That effect is conservative and shrinks with n
  (variance ratio 1.09 at n=8000). We moved from under-covering, which is the real danger, to
  slightly over-covering, which is safe.
- The numbers above are M=50 point estimates (coverage SE about 3 points). An M=200 confirm on the
  bug, oracle, and fix cells is running.
- The result is specific to the binary-treatment linear family, where the Hessian is fully
  determined by e(x). The structured-Λ idea generalizes (estimate the few scalars the curvature
  depends on, build Λ analytically), but each family needs its own derivation.
- The propensity clip at 0.001 caps the overlap weight, a mild bias at the tails. Overlap trimming
  is the principled alternative.

## Reproduce

```
PYTHONPATH=src python3.11 exploration/lambda_decomp.py --sanity      # oracle psi == closed form
PYTHONPATH=src python3.11 exploration/lambda_decomp.py --M 30        # full 11-object decomposition
PYTHONPATH=src python3.11 exploration/lambda_inv_fix.py --M 50       # the fix + estimator sweep
```

The injection hook (`lambda_eval_fn`) and the exposed `result.lambda_hat` live in
`src/deep_inference/core/algorithm.py`. Both default to the prior behavior.
