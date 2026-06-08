"""
deep_inference: Structural Deep Learning with Valid Inference

Usage:
    # Legacy API (13 GLM families)
    from deep_inference import structural_dml
    result = structural_dml(Y, T, X, family='logit')

    # New API (flexible targets + regimes)
    from deep_inference import inference
    result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

    # Results
    print(result.summary())
"""

import numpy as np
from typing import Callable, List, Optional
from torch import Tensor

from .core import structural_dml_core, DMLResult
from .families import get_family, FAMILY_REGISTRY, BaseFamily


def _validate_inputs(Y, T, X, family_name=None):
    """Validate and convert inputs. Returns (Y, T, X) as numpy arrays."""
    import warnings

    # Sparse matrix check (issue 13)
    try:
        import scipy.sparse
        if scipy.sparse.issparse(Y) or scipy.sparse.issparse(T) or scipy.sparse.issparse(X):
            raise ValueError("Sparse matrices not supported. Convert with .toarray() first.")
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
    if family_name in ('poisson', 'negbin'):
        if np.any(Y < 0):
            raise ValueError(f"Y contains negative values, invalid for family='{family_name}'.")

    if family_name in ('logit', 'probit'):
        if len(np.unique(Y)) == 1:
            warnings.warn(
                f"All Y values are {Y[0]}. Perfect separation — estimation may fail.",
                UserWarning,
            )

    if family_name in ('gamma', 'weibull'):
        if np.any(Y <= 0):
            raise ValueError(f"Y contains non-positive values, invalid for family='{family_name}'.")

    return Y, T, X


def structural_dml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    family: Optional[str] = None,
    target: Optional[str] = None,
    loss_fn: Optional[Callable] = None,
    target_fn: Optional[Callable] = None,
    theta_dim: Optional[int] = None,
    n_folds: int = 50,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 10,
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
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs per fold
        lr: Learning rate
        patience: Early stopping patience (default=10; use 50 for complex models)
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

    # Get family or use custom functions
    if family is not None:
        # Build family kwargs (e.g., target='ame' for logit, n_alternatives=3 for multinomial)
        # Known family constructor params that should NOT pass through to structural_dml_core
        FAMILY_CONSTRUCTOR_PARAMS = {'target', 'n_alternatives', 'n_attributes', 'target_idx',
                                     'shape', 'scale', 'overdispersion', 'sigma'}
        family_kwargs = {}
        if target is not None:
            family_kwargs['target'] = target
        for param in list(kwargs.keys()):
            if param in FAMILY_CONSTRUCTOR_PARAMS:
                family_kwargs[param] = kwargs.pop(param)

        fam = get_family(family, **family_kwargs)
        loss_fn = fam.loss
        theta_dim = fam.theta_dim
        three_way = fam.hessian_depends_on_theta()

        # Use closed-form functions if available
        # Create test inputs with correct dimensions
        test_theta = Tensor([[0.0] * theta_dim])
        # Determine treatment dimension for test (T may be multi-dim for multinomial)
        t_test_dim = getattr(fam, 'J', 1) * getattr(fam, 'K', 1) if hasattr(fam, 'J') else 1
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
        three_way = kwargs.pop('three_way', None)  # Auto-detect
        gradient_fn = None
        hessian_fn = None
        per_obs_target_fn = None
        per_obs_target_grad_fn = None

        # Default target if not provided
        if target_fn is None:
            def target_fn(x, theta):
                return theta[:, 1].mean()

    # Extract network_factory from kwargs if present
    network_factory = kwargs.pop('network_factory', None)

    result = structural_dml_core(
        Y=Y,
        T=T,
        X=X,
        loss_fn=loss_fn,
        target_fn=target_fn,
        theta_dim=theta_dim,
        n_folds=n_folds,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        patience=patience,
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
    _model: Optional[str] = None
    _target: Optional[str] = None
    _n_obs: Optional[int] = None
    _n_folds: Optional[int] = None

    # Fields for prediction capability
    _X_train: Optional[np.ndarray] = field(default=None, repr=False)
    _theta_predictor: Optional[Any] = field(default=None, repr=False)

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
    model: Optional[str] = None,
    target: Optional[str] = None,
    # Option 2: Custom loss/target functions
    loss: Optional[Callable] = None,
    target_fn: Optional[Callable] = None,
    theta_dim: Optional[int] = None,
    # Hessian property overrides (for custom loss with non-scalar T)
    hessian_depends_on_theta: Optional[bool] = None,
    hessian_depends_on_y: Optional[bool] = None,
    # Evaluation point
    t_tilde: Optional[float] = None,
    # Randomization settings (for Regime A)
    is_randomized: bool = False,
    treatment_dist: Optional["TreatmentDistribution"] = None,
    # Lambda estimation override
    lambda_method: Optional[str] = None,
    # Cross-fitting settings
    n_folds: int = 50,
    # Network settings
    hidden_dims: List[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    # Custom network architecture
    network_factory: Optional[Callable] = None,
    # Other
    ridge: float = 1e-4,
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
        model: Built-in model name ("linear", "logit", etc.)
        loss: Custom loss function: loss(y, t, theta) -> scalar

        # Target specification (choose one):
        target: Built-in target name ("beta", "ame", etc.)
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

        # Network:
        hidden_dims: Hidden layer sizes
        epochs: Training epochs
        lr: Learning rate

        ridge: Regularization for Lambda inversion
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
    from .models import Linear, Logit, MultinomialLogit, CombinatorialModel, Quantile, CustomModel, model_from_loss
    from .targets import AverageParameter, AME, CustomTarget, ChoiceProbabilityTarget, MultinomialAME, Elasticity, WTP, ConsumerWelfare, DoseResponse, Profit, TailProbability, ConditionalVariance, MultiTreatmentATE
    from .lambda_ import select_lambda_strategy, Regime, detect_regime
    from .engine import run_crossfit

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

    # Resolve model
    if model is not None:
        # Built-in model
        # Extract quantile-specific kwargs before building model map
        tau = kwargs.pop("tau", 0.5)
        smooth_eps = kwargs.pop("smooth_eps", 0.01)

        model_map = {
            "linear": Linear(),
            "logit": Logit(),
            "multinomial_logit": MultinomialLogit(),
            "quantile": Quantile(tau=tau, smooth_eps=smooth_eps),
        }
        if model not in model_map:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map.keys())}")
        struct_model = model_map[model]
    elif loss is not None:
        # Custom loss
        if theta_dim is None:
            raise ValueError("theta_dim required for custom loss")
        struct_model = model_from_loss(
            loss, theta_dim,
            hessian_depends_on_theta=hessian_depends_on_theta,
            hessian_depends_on_y=hessian_depends_on_y,
        )
    else:
        raise ValueError("Must provide 'model' or 'loss'")

    # Resolve target
    if target is not None:
        # Built-in target
        target_map = {
            "beta": AverageParameter(param_index=1, theta_dim=struct_model.theta_dim),
            "average_slope": AverageParameter(param_index=1, theta_dim=struct_model.theta_dim),
            "ame": AME(param_index=1, model_type="logit" if model == "logit" else "linear"),
            "elasticity": Elasticity(model_type=model if model in ("logit", "poisson", "gamma", "negbin") else "logit"),
            "wtp": WTP(attribute_index=1, price_index=2),
            "welfare": ConsumerWelfare(price_coef_index=1),
            "dose_response": DoseResponse(model_type=model if model in ("logit", "linear", "poisson") else "logit"),
            "profit": Profit(model_type=model if model in ("logit", "linear", "poisson") else "logit"),
            "tail_probability": TailProbability(model_type=model if model in ("logit", "poisson", "linear") else "logit"),
            "conditional_variance": ConditionalVariance(model_type=model if model in ("logit", "poisson") else "logit"),
            "qte": AverageParameter(param_index=1, theta_dim=struct_model.theta_dim),
        }
        if target not in target_map:
            raise ValueError(f"Unknown target: {target}. Available: {list(target_map.keys())}")
        struct_target = target_map[target]
    elif target_fn is not None:
        # Custom target
        struct_target = CustomTarget(h_fn=target_fn)
    else:
        # Default: average beta
        struct_target = AverageParameter(param_index=1, theta_dim=struct_model.theta_dim)

    # Select Lambda strategy
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
        ridge=ridge,
        verbose=verbose,
        network_factory=network_factory,
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


def did_2x2(
    Y,
    group,
    post,
    *,
    alpha: float = 0.05,
    use_bessel: bool = False,
) -> InferenceResult:
    """
    Closed-form 2x2 repeated-cross-section difference-in-differences with
    influence-function inference.

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
    from .did import did_2x2_arrays

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
        _model="did_2x2",
        _target="DID",
        _n_obs=out["n"],
        _n_folds=1,
    )


# Re-export key classes
from .core import DMLResult, compute_coverage, compute_se_ratio
from .families import (
    LinearFamily,
    LogitFamily,
    PoissonFamily,
    GammaFamily,
    GumbelFamily,
    TobitFamily,
    NegBinFamily,
    WeibullFamily,
    BaseFamily,
)

# New architecture exports
from .models import StructuralModel, CustomModel, Linear, Logit, MultinomialLogit, CombinatorialModel, Quantile
from .targets import Target, CustomTarget, AverageParameter, AME, ChoiceProbabilityTarget, MultinomialAME, Elasticity, WTP, ConsumerWelfare, DoseResponse, Profit, TailProbability, ConditionalVariance, MultiTreatmentATE
from .families import MultinomialLogitFamily
from .lambda_ import Regime, detect_regime, select_lambda_strategy

__all__ = [
    # New API
    'inference',
    'InferenceResult',
    'did_2x2',
    # Legacy API
    'structural_dml',
    # Result class
    'DMLResult',
    # Families (legacy)
    'LinearFamily',
    'LogitFamily',
    'PoissonFamily',
    'GammaFamily',
    'GumbelFamily',
    'TobitFamily',
    'NegBinFamily',
    'WeibullFamily',
    'BaseFamily',
    'get_family',
    'FAMILY_REGISTRY',
    # New architecture
    'StructuralModel',
    'CustomModel',
    'Linear',
    'Logit',
    'MultinomialLogit',
    'MultinomialLogitFamily',
    'Quantile',
    'Target',
    'CustomTarget',
    'AverageParameter',
    'AME',
    'ChoiceProbabilityTarget',
    'MultinomialAME',
    'Elasticity',
    'WTP',
    'ConsumerWelfare',
    'Regime',
    'detect_regime',
    'select_lambda_strategy',
    # Utilities
    'compute_coverage',
    'compute_se_ratio',
]
