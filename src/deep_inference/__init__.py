"""
deep_inference: Structural Deep Learning with Valid Inference

Usage:
    # Legacy API (GLM families)
    from deep_inference import structural_dml
    result = structural_dml(Y, T, X, family='logit')

    # New API (flexible targets + regimes)
    from deep_inference import inference
    result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

    # Difference-in-differences convenience wrapper
    from deep_inference import did
    result = did(Y, group, post)              # closed-form 2x2
    result = did(Y, group, post, X=X)         # heterogeneous neural DiD

    # Results
    print(result.summary())

Which entry point?
    structural_dml(...)  -- classic GLM families. Pick a `family`
        ('linear', 'logit', 'poisson', 'gamma', 'negbin', ...) and get
        E[beta(X)] (or a family-specific target). The familiar, batteries-
        included path for standard structural GLMs.

    inference(...)       -- flexible targets & regimes. Use when you need a
        custom loss, a custom or economic target (AME, elasticity, WTP,
        dose-response, profit, tail probability, conditional variance), an
        explicit randomization regime, or a DiD model. Built-in models:
        linear, logit, multinomial_logit, quantile, did, did_fe.

    did(...)             -- difference-in-differences convenience wrapper.
        Auto-selects the closed-form 2x2 ('exact'), heterogeneous neural
        DiD ('neural'), or two-way fixed-effects panel DiD ('panel_fe')
        from the arguments. A thin, friendly front-end over inference().
"""

from collections.abc import Callable
from typing import List, Optional

import numpy as np
from torch import Tensor

from .core import DMLResult, structural_dml_core
from .families import FAMILY_REGISTRY, BaseFamily, get_family


def _validate_inputs(Y, T, X, family_name=None):
    """Validate and convert inputs. Returns (Y, T, X) as numpy arrays."""
    import warnings

    # Sparse matrix check (issue 13)
    try:
        import scipy.sparse

        if (
            scipy.sparse.issparse(Y)
            or scipy.sparse.issparse(T)
            or scipy.sparse.issparse(X)
        ):
            raise ValueError(
                "Sparse matrices not supported. Convert with .toarray() first."
            )
    except ImportError:
        pass

    # Auto-convert lists, pandas, etc. (issues 11, 12)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    # Shape checks (issues 5, 6)
    if Y.ndim != 1:
        raise ValueError(f"Y must be 1D array, got shape {Y.shape}")
    if T.ndim not in (1, 2):
        raise ValueError(f"T must be 1D or 2D array, got {T.ndim}D")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")

    n = len(Y)
    if len(T) != n:
        raise ValueError(f"Y and T length mismatch: len(Y)={n}, len(T)={len(T)}")
    if len(X) != n:
        raise ValueError(f"Y and X length mismatch: len(Y)={n}, len(X)={len(X)}")

    # NaN/Inf checks (issues 3, 4)
    for name, arr in [("Y", Y), ("T", T), ("X", X)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains Inf values")

    # Constant treatment (issue 1)
    if T.ndim == 1 and np.std(T) == 0:
        raise ValueError(
            f"Treatment T is constant (all values = {T[0]}). "
            "Cannot estimate treatment effects without variation in T."
        )

    # All zeros in Y (issue 2)
    if np.all(Y == 0):
        warnings.warn("All Y values are zero. Results may be unreliable.", UserWarning)

    # Family-specific checks (issues 7, 8)
    if family_name in ("poisson", "negbin"):
        if np.any(Y < 0):
            raise ValueError(
                f"Y contains negative values, invalid for family='{family_name}'."
            )

    if family_name in ("logit", "probit"):
        if len(np.unique(Y)) == 1:
            warnings.warn(
                f"All Y values are {Y[0]}. Perfect separation — estimation may fail.",
                UserWarning,
            )

    if family_name in ("gamma", "weibull"):
        if np.any(Y <= 0):
            raise ValueError(
                f"Y contains non-positive values, invalid for family='{family_name}'."
            )

    return Y, T, X


def structural_dml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    family: str | None = None,
    target: str | None = None,
    loss_fn: Callable | None = None,
    target_fn: Callable | None = None,
    theta_dim: int | None = None,
    n_folds: int = 50,
    n_repeats: int = 1,
    three_way_theta_frac: float = 0.6,
    hidden_dims: list[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    patience: int | None = None,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    variance: str = "pooled",
    verbose: bool = False,
    store_data: bool = True,
    **kwargs,
) -> DMLResult:
    """
    Structural deep learning with valid inference.

    Implements the Farrell-Liang-Misra framework for estimating
    heterogeneous structural parameters with neural networks,
    using influence functions for valid inference.

    Requirements for Valid Inference
    --------------------------------
    1. **Model is well-specified**: The structural family matches the data
       generating process. Influence functions correct for regularization
       bias but cannot correct for model misspecification.

    2. **Network approximates θ*(x) well**: The neural network must be able
       to learn the true heterogeneity function. For simple heterogeneity,
       [64, 32] architecture is sufficient. For complex patterns, consider
       larger networks and more data.

    3. **Sufficient folds**: K >= 50 is recommended for stable SE estimation.
       With K=20, SE may be overestimated by 3x or more.

    4. **Well-conditioned Λ(x)**: Check diagnostics.min_lambda_eigenvalue.
       Near-singular Hessians can cause unstable estimates.

    If these conditions fail, coverage may be below 95%.

    Args:
        Y: (n,) outcome vector
        T: (n,) treatment vector
        X: (n, d) covariate matrix
        family: Pre-built family name. Available families:
                - 'linear': Y = alpha + beta*T + eps
                - 'logit': P(Y=1) = sigmoid(alpha + beta*T)
                - 'poisson': Y ~ Poisson(exp(alpha + beta*T))
                - 'gamma': Y ~ Gamma(shape, exp(alpha + beta*T))
                - 'gumbel': Y ~ Gumbel(alpha + beta*T, scale)
                - 'tobit': Y = max(0, alpha + beta*T + sigma*eps)
                - 'negbin': Y ~ NegBin(exp(alpha + beta*T), r)
                - 'weibull': Y ~ Weibull(shape, exp(alpha + beta*T))
                - 'gaussian': Y ~ N(alpha + beta*T, sigma(X))
                - 'probit': P(Y=1) = Phi(alpha + beta*T)
                - 'beta': Y ~ Beta(mu*phi, (1-mu)*phi), logit link
                - 'zip': Y ~ ZIP(lambda, pi), zero-inflated Poisson
                - 'multinomial_logit': P(Y=j) = softmax(α_j + x'_j·β)
        target: Target functional for inference (family-specific):
                - logit: 'beta' (log-odds, default) or 'ame' (average marginal effect)
                - tobit: 'latent' (effect on Y*, default) or 'observed' (effect on E[Y])
        loss_fn: Custom loss function (y, t, theta) -> (n,) losses
        target_fn: Custom target function (x, theta) -> scalar
        theta_dim: Dimension of parameter vector (required if custom loss)
        n_folds: Number of cross-fitting folds (default=50, minimum recommended)
        n_repeats: Repeated cross-fitting splits (default 1 == single split,
            byte-identical to before). n_repeats > 1 aggregates with the
            Chernozhukov et al. (2018) median DML rule, widening the SE by the
            across-split variation (the under-coverage fix).
        three_way_theta_frac: Fraction of training data used to fit theta in the
            3-way split (default 0.6 == previous hardcoded value).
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs per fold
        lr: Learning rate
        dropout: StructuralNet dropout (default 0.1 == previous hardcoded value).
        weight_decay: Adam L2 penalty (default 0.0 == no penalty, unchanged).
        patience: Early stopping patience. Default None resolves to 10 for all
            families except 'multinomial_logit', which auto-bumps to 50 (the
            3-way split makes patience=10 fatal — see MEMORY). Pass an explicit
            int to override.
        variance: Variance estimator for the SE/CI. 'pooled' (default) is the
            FLM influence-function variance (Bessel sample variance of psi,
            centered at the global mean). 'within_fold' is the legacy
            per-fold-centered variant (mean of per-fold variances); it omits the
            between-fold component, so its SE is always <= pooled.
        verbose: Print progress
        store_data: Store X for prediction methods (default=True)
        **kwargs: Additional arguments to structural_dml_core

    Returns:
        DMLResult with:
            - mu_hat: Point estimate
            - se: Standard error
            - ci_lower, ci_upper: 95% confidence interval
            - psi_values: Influence function values
            - theta_hat: Estimated parameters for all observations
            - diagnostics: Training and estimation diagnostics including:
                - min_lambda_eigenvalue: Check for near-singular Hessians
                - n_regularized: Count of observations needing extra regularization
                - correction_ratio: If > 2, consider more folds

    Warnings:
        - High Lambda regularization rate: Indicates numerical instability
        - High correction variance ratio: Suggests too few folds (K < 50)

    Examples:
        # Binary outcome with heterogeneous effects
        result = structural_dml(Y, T, X, family='logit')

        # Continuous outcome
        result = structural_dml(Y, T, X, family='linear')

        # Check diagnostics
        print(f"Min eigenvalue: {result.diagnostics['min_lambda_eigenvalue']:.6f}")
        print(f"Observations regularized: {result.diagnostics['n_regularized']}")

        # Custom structural model
        def tobit_loss(y, t, theta):
            import torch
            alpha, beta = theta[:, 0], theta[:, 1]
            mu = alpha + beta * t
            sigma = 1.0
            # Tobit log-likelihood
            censored = (y <= 0).float()
            uncensored = 1 - censored
            z = -mu / sigma
            ll = censored * torch.distributions.Normal(0, 1).cdf(z).log()
            ll += uncensored * (-0.5 * ((y - mu) / sigma) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sigma))
            return -ll

        result = structural_dml(Y, T, X, loss_fn=tobit_loss, theta_dim=2)
    """
    # Validate inputs
    if family is None and loss_fn is None:
        raise ValueError("Must provide either 'family' or 'loss_fn'")

    if family is not None and loss_fn is not None:
        raise ValueError("Cannot provide both 'family' and 'loss_fn'")

    if loss_fn is not None and theta_dim is None:
        raise ValueError("Must provide 'theta_dim' when using custom loss_fn")

    # Data validation and conversion
    Y, T, X = _validate_inputs(Y, T, X, family_name=family)

    # Hyperparameter validation
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    # Resolve patience. Default stays 10 for every family except multinomial_logit,
    # whose 3-way split yields noisy val loss that triggers early stopping ~15 epochs
    # at patience=10 (fatal); bump to 50 unless the caller set patience explicitly.
    if patience is None:
        patience = 50 if family == "multinomial_logit" else 10

    # Get family or use custom functions
    if family is not None:
        # Build family kwargs (e.g., target='ame' for logit, n_alternatives=3 for multinomial)
        # Known family constructor params that should NOT pass through to structural_dml_core
        FAMILY_CONSTRUCTOR_PARAMS = {
            "target",
            "n_alternatives",
            "n_attributes",
            "target_idx",
            "shape",
            "scale",
            "overdispersion",
            "sigma",
        }
        family_kwargs = {}
        if target is not None:
            family_kwargs["target"] = target
        for param in list(kwargs.keys()):
            if param in FAMILY_CONSTRUCTOR_PARAMS:
                family_kwargs[param] = kwargs.pop(param)

        fam = get_family(family, **family_kwargs)
        loss_fn = fam.loss
        theta_dim = fam.theta_dim
        # Explicit override wins (as in automatic mode); else default to the
        # family's theta-dependence. NOTE: theta-independence (e.g. linear) does
        # NOT imply Lambda is x-independent: Lambda(x)=E[l_theta_theta|x] still
        # varies in x through the treatment design E[T|x]. Forcing three_way=True
        # lets the Lambda(x) estimator capture that heterogeneity.
        three_way = kwargs.pop("three_way", None)
        if three_way is None:
            three_way = fam.hessian_depends_on_theta()

        # Use closed-form functions if available
        # Create test inputs with correct dimensions
        test_theta = Tensor([[0.0] * theta_dim])
        # Determine treatment dimension for test (T may be multi-dim for multinomial)
        t_test_dim = (
            getattr(fam, "J", 1) * getattr(fam, "K", 1) if hasattr(fam, "J") else 1
        )
        test_t = Tensor([0.0] * t_test_dim) if t_test_dim > 1 else Tensor([0.0])
        try:
            grad_result = fam.gradient(Tensor([0.0]), test_t.unsqueeze(0), test_theta)
            gradient_fn = fam.gradient if grad_result is not None else None
        except Exception:
            gradient_fn = None

        try:
            hess_result = fam.hessian(Tensor([0.0]), test_t.unsqueeze(0), test_theta)
            hessian_fn = fam.hessian if hess_result is not None else None
        except Exception:
            hessian_fn = None

        # Use family's target functions
        target_fn = fam.default_target
        per_obs_target_fn = fam.per_obs_target
        per_obs_target_grad_fn = fam.per_obs_target_gradient
    else:
        # Fully automatic mode
        three_way = kwargs.pop("three_way", None)  # Auto-detect
        gradient_fn = None
        hessian_fn = None
        per_obs_target_fn = None
        per_obs_target_grad_fn = None

        # Default target if not provided
        if target_fn is None:

            def target_fn(x, theta):
                return theta[:, 1].mean()

    # Extract network_factory from kwargs if present
    network_factory = kwargs.pop("network_factory", None)

    result = structural_dml_core(
        Y=Y,
        T=T,
        X=X,
        loss_fn=loss_fn,
        target_fn=target_fn,
        theta_dim=theta_dim,
        n_folds=n_folds,
        n_repeats=n_repeats,
        three_way_theta_frac=three_way_theta_frac,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        patience=patience,
        dropout=dropout,
        weight_decay=weight_decay,
        variance=variance,
        three_way=three_way,
        gradient_fn=gradient_fn,
        hessian_fn=hessian_fn,
        per_obs_target_fn=per_obs_target_fn,
        per_obs_target_grad_fn=per_obs_target_grad_fn,
        verbose=verbose,
        network_factory=network_factory,
        **kwargs,
    )

    # Set metadata on result
    result._family = family
    result._target = target if target else "E[beta]"
    result._n_obs = len(Y)
    result._n_folds = n_folds

    # Store X for prediction capability
    if store_data:
        result._X_train = np.asarray(X).copy()

    return result


# New API: inference() with general loss/target
from dataclasses import dataclass, field
from typing import Any

import torch

from .utils.result_mixin import PredictVisualizeMixin


@dataclass
class InferenceResult(PredictVisualizeMixin):
    """Result from the new inference() API."""

    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    psi_values: Tensor
    theta_hat: Tensor
    diagnostics: dict

    # Metadata fields
    _model: str | None = None
    _target: str | None = None
    _n_obs: int | None = None
    _n_folds: int | None = None

    # Fields for prediction capability
    _X_train: np.ndarray | None = field(default=None, repr=False)
    _theta_predictor: Any | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Short representation."""
        from .utils.formatting import format_short_repr

        return format_short_repr(
            class_name="InferenceResult",
            estimate=self.mu_hat,
            se=self.se,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
        )

    def summary(self) -> str:
        """
        Generate statsmodels-style summary.

        Returns:
            Formatted summary string
        """
        from .utils.formatting import format_full_summary

        # Determine target name for display
        target_name = self._target if self._target else "E[beta]"

        return format_full_summary(
            title="Structural Inference Results",
            coef_name=target_name,
            estimate=self.mu_hat,
            se=self.se,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
            diagnostics=self.diagnostics,
            family=self._model,
            target=target_name,
            n_obs=self._n_obs,
            n_folds=self._n_folds,
        )


def inference(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    # Option 1: Built-in model/target (strings)
    model: str | None = None,
    target: str | None = None,
    # Option 2: Custom loss/target functions
    loss: Callable | None = None,
    target_fn: Callable | None = None,
    theta_dim: int | None = None,
    # Hessian property overrides (for custom loss with non-scalar T)
    hessian_depends_on_theta: bool | None = None,
    hessian_depends_on_y: bool | None = None,
    # Evaluation point
    t_tilde: float | None = None,
    # Randomization settings (for Regime A)
    is_randomized: bool = False,
    treatment_dist: Optional["TreatmentDistribution"] = None,
    # Lambda estimation override
    lambda_method: str | None = None,
    # Advanced: supply a pre-built LambdaStrategy directly (bypasses auto-selection).
    # Used to inject a custom/oracle Λ(x); the clean replacement for the legacy
    # structural_dml lambda_eval_fn hook, which the new inference() path lacks.
    lambda_strategy: Optional["LambdaStrategy"] = None,
    # Cross-fitting settings
    n_folds: int = 50,
    n_repeats: int = 1,
    three_way_theta_frac: float = 0.6,
    # Network settings
    hidden_dims: list[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    # Custom network architecture
    network_factory: Callable | None = None,
    # Other
    tikhonov_scale: float = 0.01,
    variance: str = "pooled",
    verbose: bool = False,
    store_data: bool = True,
    # Model-specific kwargs (e.g., tau, smooth_eps for quantile)
    **kwargs,
) -> InferenceResult:
    """
    General inference with user-provided loss and target.

    This is the new API that supports arbitrary loss functions and targets.
    Everything is derived via autodiff unless closed-forms are provided.

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates

        # Model specification (choose one):
        model: Built-in model name. One of:
            "linear", "logit", "multinomial_logit", "quantile", "did", "did_fe".
            (Most users reach "did"/"did_fe" via the did() wrapper.)
        loss: Custom loss function: loss(y, t, theta) -> scalar

        # Target specification (choose one):
        target: Built-in target name. One of:
            "beta" (E[beta_1], default), "average_slope", "ame",
            "elasticity", "wtp", "welfare", "dose_response", "profit",
            "tail_probability", "conditional_variance", "qte",
            "tau"/"att" (saturated DiD interaction, theta[3]),
            "fe_effect" (FE panel DiD effect, theta[0]).
        target_fn: Custom target: h(x, theta, t_tilde) -> scalar

        theta_dim: Parameter dimension (required for custom loss)
        hessian_depends_on_theta: Override auto-detection for custom loss
            (needed when T is multi-dimensional and auto-detection fails)
        hessian_depends_on_y: Override auto-detection for custom loss
        t_tilde: Evaluation point (default: mean(T))

        # Regime settings:
        is_randomized: True if T is randomly assigned
        treatment_dist: Distribution F_T (enables Regime A computation)
        lambda_method: Override auto-detection ("compute", "analytic", "estimate")

        # Cross-fitting:
        n_folds: Number of folds (default: 50)
        n_repeats: Number of repeated cross-fitting splits (default 1 == single
            split, byte-identical to before). For n_repeats > 1 the estimate/SE
            are aggregated with the Chernozhukov et al. (2018) median DML rule
            (mu_hat = median_r(mu_r), se^2 = median_r(se_r^2 + (mu_r-mu_hat)^2)),
            which widens the SE by the across-split variation — the under-coverage
            fix. psi/theta are kept from the first repeat.
        three_way_theta_frac: Fraction of training data used to fit theta in the
            3-way (Regime C) split (default 0.6 == previous hardcoded value); an
            undersmoothing knob.

        # Network:
        hidden_dims: Hidden layer sizes
        epochs: Training epochs
        lr: Learning rate
        dropout: StructuralNet dropout (default 0.1 == previous hardcoded value;
            ignored when network_factory is supplied). An undersmoothing knob.
        weight_decay: Adam L2 penalty (default 0.0 == no penalty, unchanged).

        tikhonov_scale: Scale ε for the Tikhonov Lambda inversion (Λ + εI)⁻¹
            with ε = tikhonov_scale·trace(Λ)/d (default 0.01).
        variance: Variance estimator for the SE/CI. 'pooled' (default) is the
            FLM influence-function variance (Bessel sample variance of psi,
            centered at the global mean). 'within_fold' is the legacy
            per-fold-centered variant (always <= pooled).
        verbose: Print progress
        store_data: Store X for prediction methods (default=True)

    Returns:
        InferenceResult with mu_hat, se, ci, psi_values, theta_hat, diagnostics

    Examples:
        # Built-in model and target
        result = inference(Y, T, X, model="logit", target="ame")

        # Custom loss and target
        def my_loss(y, t, theta):
            p = torch.sigmoid(theta[0] + theta[1] * t)
            return -y * torch.log(p) - (1-y) * torch.log(1-p)

        def my_target(x, theta, t_tilde):
            p = torch.sigmoid(theta[0] + theta[1] * t_tilde)
            return p * (1-p) * theta[1]

        result = inference(Y, T, X, loss=my_loss, target_fn=my_target, theta_dim=2)
    """
    from .engine import run_crossfit
    from .lambda_ import Regime, detect_regime, select_lambda_strategy
    from .models import (
        CombinatorialModel,
        CustomModel,
        DiDModel,
        FEPanelDiDModel,
        Linear,
        Logit,
        MultinomialLogit,
        Quantile,
        model_from_loss,
    )
    from .targets import (
        AME,
        WTP,
        AverageParameter,
        ChoiceProbabilityTarget,
        ConditionalVariance,
        ConsumerWelfare,
        CustomTarget,
        DoseResponse,
        Elasticity,
        MultinomialAME,
        MultiTreatmentATE,
        Profit,
        TailProbability,
    )

    # Data validation and conversion
    Y, T, X = _validate_inputs(Y, T, X, family_name=model)

    # Convert inputs to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Default t_tilde to mean treatment
    if t_tilde is None:
        t_tilde = T_t.mean()
    else:
        t_tilde = torch.tensor(t_tilde, dtype=torch.float32)

    # Index of the slope coefficient beta_1 within theta (default [alpha, beta]).
    # Overridden below for multinomial logit, whose layout is
    # [alpha_1, ..., alpha_{J-1}, beta_1, ..., beta_K].
    beta_index = 1

    # Resolve model
    if model is not None:
        # Built-in model
        # Extract model-specific kwargs before building model map
        tau = kwargs.pop("tau", 0.5)
        smooth_eps = kwargs.pop("smooth_eps", 0.01)
        # Multinomial logit dimensions (ignored by the other models)
        n_alternatives = kwargs.pop("n_alternatives", 3)
        n_attributes = kwargs.pop("n_attributes", 2)
        target_idx = kwargs.pop("target_idx", 0)

        model_map = {
            "linear": Linear(),
            "logit": Logit(),
            "multinomial_logit": MultinomialLogit(
                n_alternatives=n_alternatives, n_attributes=n_attributes
            ),
            "quantile": Quantile(tau=tau, smooth_eps=smooth_eps),
            "did": DiDModel(),
            "did_fe": FEPanelDiDModel(),
        }
        if model not in model_map:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(model_map.keys())}"
            )
        struct_model = model_map[model]

        # Multinomial theta is [alpha_1..alpha_{J-1}, beta_1..beta_K]; the first
        # attribute coefficient beta_1 lives at index (J-1), NOT index 1.
        if model == "multinomial_logit":
            beta_index = (n_alternatives - 1) + target_idx
    elif loss is not None:
        # Custom loss
        if theta_dim is None:
            raise ValueError("theta_dim required for custom loss")
        struct_model = model_from_loss(
            loss,
            theta_dim,
            hessian_depends_on_theta=hessian_depends_on_theta,
            hessian_depends_on_y=hessian_depends_on_y,
        )
    else:
        raise ValueError("Must provide 'model' or 'loss'")

    # Fail loudly on silently-dropped arguments (nothing downstream consumes kwargs).
    if kwargs:
        raise ValueError(
            f"inference() got unexpected keyword arguments: {list(kwargs)}"
        )

    # Resolve target
    if target is not None:
        # Built-in target
        target_map = {
            "beta": AverageParameter(
                param_index=beta_index, theta_dim=struct_model.theta_dim
            ),
            "average_slope": AverageParameter(
                param_index=1, theta_dim=struct_model.theta_dim
            ),
            "ame": AME(
                param_index=1, model_type="logit" if model == "logit" else "linear"
            ),
            "elasticity": Elasticity(model_type="logit"),
            "wtp": WTP(attribute_index=1, price_index=2),
            "welfare": ConsumerWelfare(price_coef_index=1),
            "dose_response": DoseResponse(
                model_type=model if model in ("logit", "linear") else "logit"
            ),
            "profit": Profit(
                model_type=model if model in ("logit", "linear") else "logit"
            ),
            "tail_probability": TailProbability(
                model_type=model if model in ("logit", "linear") else "logit"
            ),
            "conditional_variance": ConditionalVariance(model_type="logit"),
            "qte": AverageParameter(param_index=1, theta_dim=struct_model.theta_dim),
            # Saturated DiD interaction tau = theta[3] (E[tau(X)] DiD effect)
            "tau": AverageParameter(param_index=3, theta_dim=struct_model.theta_dim),
            "att": AverageParameter(param_index=3, theta_dim=struct_model.theta_dim),
            # FE panel DiD effect tau = theta[0] (theta_dim=1, intercept-free)
            "fe_effect": AverageParameter(
                param_index=0, theta_dim=struct_model.theta_dim
            ),
        }
        if target not in target_map:
            raise ValueError(
                f"Unknown target: {target}. Available: {list(target_map.keys())}"
            )
        struct_target = target_map[target]
    elif target_fn is not None:
        # Custom target
        struct_target = CustomTarget(h_fn=target_fn)
    else:
        # Default: average beta
        struct_target = AverageParameter(
            param_index=beta_index, theta_dim=struct_model.theta_dim
        )

    # Select Lambda strategy (unless the caller supplied one, e.g. an oracle Λ(x))
    if lambda_strategy is None:
        lambda_strategy = select_lambda_strategy(
            model=struct_model,
            is_randomized=is_randomized,
            treatment_dist=treatment_dist,
            lambda_method=lambda_method,
        )

    # Detect regime for diagnostics
    regime = detect_regime(struct_model, is_randomized, treatment_dist is not None)

    if verbose:
        from .lambda_.selector import describe_regime

        print(f"Detected: {describe_regime(regime)}")

    # Run cross-fitting
    result = run_crossfit(
        Y=Y_t,
        T=T_t,
        X=X_t,
        t_tilde=t_tilde,
        model=struct_model,
        target=struct_target,
        lambda_strategy=lambda_strategy,
        n_folds=n_folds,
        epochs=epochs,
        lr=lr,
        patience=patience,
        hidden_dims=hidden_dims,
        tikhonov_scale=tikhonov_scale,
        variance=variance,
        verbose=verbose,
        network_factory=network_factory,
        n_repeats=n_repeats,
        three_way_theta_frac=three_way_theta_frac,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    inf_result = InferenceResult(
        mu_hat=result.mu_hat,
        se=result.se,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        psi_values=result.psi_values,
        theta_hat=result.theta_hat,
        diagnostics={
            "regime": regime.name,
            "n_folds": n_folds,
            "lambda_method": lambda_strategy.__class__.__name__,
        },
        _model=model,
        _target=target if target else "E[beta]",
        _n_obs=len(Y),
        _n_folds=n_folds,
    )

    # Store X for prediction capability
    if store_data:
        inf_result._X_train = np.asarray(X).copy()

    return inf_result


def _did_exact(
    Y,
    group,
    post,
    *,
    alpha: float = 0.05,
    use_bessel: bool = False,
) -> InferenceResult:
    """
    Closed-form 2x2 repeated-cross-section difference-in-differences with
    influence-function inference. Internal helper for did(method='exact').

    Estimand: the group x post interaction beta = mu_11 - mu_10 - mu_01 + mu_00.
    This is a design-based closed-form estimator (no neural net / cross-fitting),
    but it follows the package's psi -> mean -> SE convention, so the returned
    standard error equals the saturated-OLS HC0 robust SE to machine precision.

    Args:
        Y: Outcome, 1-D array of length n.
        group: Binary treatment-group indicator G in {0, 1}, length n.
        post: Binary post-period indicator T in {0, 1}, length n.
        alpha: CI level (default 0.05 -> 95% CI).
        use_bessel: If False (default), use the n denominator (matches HC0). If True,
            apply the (n-1) finite-sample correction (will NOT equal HC0 exactly).

    Returns:
        InferenceResult with mu_hat (beta), se, ci_lower, ci_upper, psi_values,
        theta_hat (cell means), and diagnostics.
    """
    from ._did_closed import did_2x2_arrays

    out = did_2x2_arrays(
        Y=Y,
        group=group,
        post=post,
        alpha=alpha,
        use_bessel=use_bessel,
    )

    return InferenceResult(
        mu_hat=out["mu_hat"],
        se=out["se"],
        ci_lower=out["ci_lower"],
        ci_upper=out["ci_upper"],
        psi_values=out["psi_values"],
        theta_hat=out["theta_hat"],
        diagnostics=out["diagnostics"],
        _model="did(exact)",
        _target="DID",
        _n_obs=out["n"],
        _n_folds=1,
    )


def _did_neural(
    Y,
    group,
    post,
    X,
    *,
    hidden_dims: list[int] | None = None,
    n_folds: int = 50,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    lambda_method: str | None = None,
    verbose: bool = False,
) -> InferenceResult:
    """
    Heterogeneous neural 2x2 difference-in-differences. Internal helper for
    did(method='neural').

    Fits the saturated DiD regression with covariate-varying coefficients via a neural
    network, Y = alpha(X) + gamma(X) G + lambda(X) P + tau(X) (G P) + eps, and returns
    the influence-function estimate of the average DiD effect E[tau(X)] with a valid
    standard error.

    This is the heterogeneous counterpart of did_2x2(): it uses the structural network +
    cross-fitting + IF machinery (Regime B, analytic Lambda = E[W W' | X], two-way split).
    Group and period assignment is assumed independent of X (the canonical repeated
    cross-section design); if cell shares depend on X, pass lambda_method='estimate'.

    Args:
        Y: Outcome, 1-D array of length n.
        group: Binary treatment-group indicator G in {0, 1}, length n.
        post: Binary post-period indicator T in {0, 1}, length n.
        X: Covariates (n, d_x) driving heterogeneity.
        hidden_dims: Network architecture (default [64, 32]).
        n_folds, epochs, lr, patience: Cross-fitting / training settings.
        lambda_method: Override Lambda estimation (default: analytic aggregate, Regime B).
        verbose: Print regime/progress.

    Returns:
        InferenceResult for E[tau(X)] (mu_hat, se, ci, psi_values, theta_hat, diagnostics).
        theta_hat columns are [alpha, gamma, lambda, tau]; use
        result.predict_theta(X_new)[:, 3] for the conditional DiD effect tau(X_new).
    """
    from ._did_closed import _as_binary_1d

    Y = np.asarray(Y, dtype=np.float64)
    G = _as_binary_1d(group, "group").astype(np.float64)
    P = _as_binary_1d(post, "post").astype(np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    T_did = np.column_stack([G, P, G * P])

    result = inference(
        Y=Y,
        T=T_did,
        X=X,
        model="did",
        target="tau",
        t_tilde=0.0,  # target ignores t_tilde; avoids the multi-col default
        n_folds=n_folds,
        epochs=epochs,
        lr=lr,
        patience=patience,
        hidden_dims=hidden_dims if hidden_dims is not None else [64, 32],
        lambda_method=lambda_method,
        verbose=verbose,
    )
    result._model = "did(neural)"
    result._target = "E[tau(X)]"
    return result


def _did_panel_fe(
    Y,
    D,
    X,
    unit,
    time,
    *,
    hidden_dims: list[int] | None = None,
    n_folds: int = 50,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    lambda_method: str | None = None,
    verbose: bool = False,
) -> InferenceResult:
    """
    Heterogeneous two-way fixed-effects panel difference-in-differences. Internal
    helper for did(method='panel_fe').

    Fits the within-transformed FE DiD model: after residualizing the outcome Y and
    treatment D by unit and time fixed effects, Ytilde_it = Dtilde_it * tau(X_it) + eps,
    and returns the influence-function estimate of the average effect E[tau(X)] with a
    valid standard error. Runs in Regime B (analytic Lambda = E[Dtilde^2 | X], two-way
    cross-fitting).

    Works for both continuous and binary outcomes. For a binary Y this is a
    fixed-effects LINEAR PROBABILITY model; the IF standard error is
    heteroskedasticity-robust, so the binary variance is handled correctly.

    Args:
        Y: Outcome, 1-D array of length n = (#units x #periods, stacked any order).
        D: Treatment indicator (e.g. D = G * Post for the canonical 2x2), length n.
        X: Covariates (n, d_x) driving effect heterogeneity (NOT demeaned).
        unit: Unit identifiers (n,). time: Period identifiers (n,).
        hidden_dims, n_folds, epochs, lr, patience: training / cross-fitting settings.
        lambda_method: Override Lambda estimation (default: intercept-free analytic, Regime B).
        verbose: Print regime/progress.

    Returns:
        InferenceResult for E[tau(X)] (mu_hat, se, ci, psi_values, theta_hat, diagnostics).
        theta_hat is (n, 1) = tau(X); use result.predict_theta(X_new)[:, 0] for tau(X_new).

    Note:
        The standard error assumes errors are independent across observations. Serial /
        within-unit correlation would require cluster-robust SEs (not implemented here).
    """
    from .utils import residualize_fixed_effects

    Y = np.asarray(Y, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    Y_tilde = residualize_fixed_effects(Y, unit, time)
    D_tilde = residualize_fixed_effects(D, unit, time)

    result = inference(
        Y=Y_tilde,
        T=D_tilde,
        X=X,
        model="did_fe",
        target="fe_effect",
        t_tilde=0.0,  # target ignores t_tilde
        n_folds=n_folds,
        epochs=epochs,
        lr=lr,
        patience=patience,
        hidden_dims=hidden_dims if hidden_dims is not None else [64, 32],
        lambda_method=lambda_method,
        verbose=verbose,
    )
    result._model = "did(panel_fe)"
    result._target = "E[tau(X)]"
    return result


def did(
    Y,
    group=None,
    post=None,
    X=None,
    *,
    D=None,
    unit=None,
    time=None,
    method: str = "auto",
    alpha: float = 0.05,
    use_bessel: bool = False,
    hidden_dims: list[int] | None = None,
    n_folds: int = 50,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    lambda_method: str | None = None,
    verbose: bool = False,
) -> InferenceResult:
    """
    Difference-in-differences with influence-function inference — single entry point.

    Three estimators, auto-selected from the arguments (override with ``method=``):

    - ``method='exact'``   — ``did(Y, group, post)``
        Closed-form 2x2 repeated cross-section. Estimand beta = mu11-mu10-mu01+mu00;
        SE equals saturated-OLS HC0 to machine precision. No covariates, no network.
    - ``method='neural'``  — ``did(Y, group, post, X=X)``
        Heterogeneous saturated 2x2: a network learns alpha/gamma/lambda/tau(X);
        target E[tau(X)]. Regime B, two-way cross-fitting.
    - ``method='panel_fe'`` — ``did(Y, group, post, X=X, unit=u, time=t)`` (or ``D=...``)
        Two-way fixed-effects panel DiD. Y and D = group*post (or an explicit D) are
        residualized by unit+time fixed effects, then a network learns tau(X); target
        E[tau(X)]. Works for continuous and binary (linear probability model) outcomes.

    Auto-selection (``method='auto'``): ``unit`` and ``time`` given -> 'panel_fe';
    else ``X`` given -> 'neural'; else -> 'exact'.

    Args:
        Y: Outcome (n,).
        group, post: Binary indicators (n,). For panel_fe, D = group*post unless D given.
        X: Covariates (n, d_x). Required for 'neural' and 'panel_fe'.
        D: Panel treatment indicator (n,), used by 'panel_fe' instead of group*post.
        unit, time: Panel identifiers (n,), required for 'panel_fe'.
        method: 'auto' | 'exact' | 'neural' | 'panel_fe'.
        alpha, use_bessel: closed-form ('exact') options.
        hidden_dims, n_folds, epochs, lr, patience, lambda_method, verbose: neural/panel options.

    Returns:
        InferenceResult (mu_hat, se, ci_lower, ci_upper, psi_values, theta_hat, diagnostics).
        For 'neural', theta_hat columns are [alpha, gamma, lambda, tau] (tau at index 3);
        for 'panel_fe', theta_hat is (n, 1) = tau(X).
    """
    if method == "auto":
        if unit is not None and time is not None:
            method = "panel_fe"
        elif X is not None:
            method = "neural"
        else:
            method = "exact"

    if method == "exact":
        if group is None or post is None:
            raise ValueError("did(method='exact') requires `group` and `post`.")
        return _did_exact(Y, group, post, alpha=alpha, use_bessel=use_bessel)

    if method == "neural":
        if group is None or post is None or X is None:
            raise ValueError("did(method='neural') requires `group`, `post`, and `X`.")
        return _did_neural(
            Y,
            group,
            post,
            X,
            hidden_dims=hidden_dims,
            n_folds=n_folds,
            epochs=epochs,
            lr=lr,
            patience=patience,
            lambda_method=lambda_method,
            verbose=verbose,
        )

    if method == "panel_fe":
        if unit is None or time is None or X is None:
            raise ValueError("did(method='panel_fe') requires `X`, `unit`, and `time`.")
        if D is None:
            if group is None or post is None:
                raise ValueError(
                    "did(method='panel_fe') requires `D` (or `group` and `post` to form "
                    "D=group*post)."
                )
            D = np.asarray(group, dtype=np.float64) * np.asarray(post, dtype=np.float64)
        return _did_panel_fe(
            Y,
            D,
            X,
            unit,
            time,
            hidden_dims=hidden_dims,
            n_folds=n_folds,
            epochs=epochs,
            lr=lr,
            patience=patience,
            lambda_method=lambda_method,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown method: {method!r}. Use 'auto', 'exact', 'neural', or 'panel_fe'."
    )


# Re-export key classes
from .core import DMLResult, compute_coverage, compute_se_ratio
from .families import (
    GammaFamily,
    GumbelFamily,
    LinearFamily,
    LogitFamily,
    MultinomialLogitFamily,
    NegBinFamily,
    PoissonFamily,
    TobitFamily,
    WeibullFamily,
)
from .lambda_ import Regime, detect_regime, select_lambda_strategy

# New architecture exports
from .models import (
    CombinatorialModel,
    CustomModel,
    DiDModel,
    FEPanelDiDModel,
    Linear,
    Logit,
    MultinomialLogit,
    Quantile,
    StructuralModel,
)
from .riesz import RieszNet, riesz_inference
from .targets import (
    AME,
    WTP,
    AverageParameter,
    ChoiceProbabilityTarget,
    ConditionalVariance,
    ConsumerWelfare,
    CustomTarget,
    DoseResponse,
    Elasticity,
    MultinomialAME,
    MultiTreatmentATE,
    Profit,
    TailProbability,
    Target,
)

__all__ = [
    # New API
    "inference",
    "InferenceResult",
    "riesz_inference",
    "RieszNet",
    "did",
    # Legacy API
    "structural_dml",
    # Result class
    "DMLResult",
    # Families (legacy)
    "LinearFamily",
    "LogitFamily",
    "PoissonFamily",
    "GammaFamily",
    "GumbelFamily",
    "TobitFamily",
    "NegBinFamily",
    "WeibullFamily",
    "BaseFamily",
    "get_family",
    "FAMILY_REGISTRY",
    # New architecture
    "StructuralModel",
    "CustomModel",
    "Linear",
    "Logit",
    "MultinomialLogit",
    "MultinomialLogitFamily",
    "Quantile",
    "CombinatorialModel",
    "DiDModel",
    "FEPanelDiDModel",
    "Target",
    "CustomTarget",
    "AverageParameter",
    "AME",
    "ChoiceProbabilityTarget",
    "MultinomialAME",
    "Elasticity",
    "WTP",
    "ConsumerWelfare",
    "DoseResponse",
    "Profit",
    "TailProbability",
    "ConditionalVariance",
    "MultiTreatmentATE",
    "Regime",
    "detect_regime",
    "select_lambda_strategy",
    # Utilities
    "compute_coverage",
    "compute_se_ratio",
]
