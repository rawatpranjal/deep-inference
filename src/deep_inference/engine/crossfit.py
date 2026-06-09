"""
Cross-fitting engine.

Orchestrates 2-way or 3-way cross-fitting based on Lambda regime.

2-way (Regimes A, B):
    For fold k:
        Train: fit θ̂, compute/fit Λ̂
        Test: evaluate ψ

3-way (Regime C):
    For fold k:
        Fold A: fit θ̂
        Fold B: compute Hessians, fit Λ̂
        Test: evaluate ψ
"""

from typing import TYPE_CHECKING, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor

# Optional tqdm import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel
    from deep_inference.targets import Target
    from deep_inference.lambda_ import LambdaStrategy


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_idx: int
    psi_values: Tensor
    theta_hat: Tensor
    eval_indices: List[int]
    train_history: Optional[dict] = None


@dataclass
class CrossFitResult:
    """Complete cross-fitting results."""

    psi_values: Tensor  # (n,) all ψ values
    theta_hat: Tensor  # (n, d_theta) all θ̂ values
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    fold_results: List[FoldResult]

    def __repr__(self) -> str:
        """Short representation."""
        return (
            f"<CrossFitResult: mu_hat={self.mu_hat:.4f}, se={self.se:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]>"
        )

    def summary(self) -> str:
        """Generate cross-fitting summary."""
        n_folds = len(self.fold_results)
        n_obs = len(self.psi_values)

        lines = [
            "Cross-Fitting Results:",
            f"  Number of folds:    {n_folds}",
            f"  Observations:       {n_obs}",
            f"  Point estimate:     {self.mu_hat:.6f}",
            f"  Standard error:     {self.se:.6f}",
            f"  95% CI:             [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
        ]

        # Add fold-level info if available
        if self.fold_results and self.fold_results[0].train_history:
            val_losses = [
                fr.train_history.get("val_loss", float("nan"))
                for fr in self.fold_results
                if fr.train_history
            ]
            if val_losses:
                mean_val = np.mean(val_losses)
                lines.append(f"  Mean val loss:      {mean_val:.6f}")

        return "\n".join(lines)


class CrossFitter:
    """
    Cross-fitting orchestrator.

    Handles both 2-way and 3-way splitting based on Lambda strategy.
    """

    def __init__(
        self,
        n_folds: int = 50,
        three_way_theta_frac: float = 0.6,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize cross-fitter.

        Args:
            n_folds: Number of folds (K)
            three_way_theta_frac: Fraction for theta training in 3-way split
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.three_way_theta_frac = three_way_theta_frac
        self.shuffle = shuffle
        self.random_state = random_state

    def create_folds(self, n: int) -> List[np.ndarray]:
        """
        Create fold indices.

        Args:
            n: Total number of observations

        Returns:
            List of arrays, each containing indices for one fold
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        # Split into K folds
        fold_size = n // self.n_folds
        folds = []

        for k in range(self.n_folds):
            start = k * fold_size
            if k == self.n_folds - 1:
                # Last fold gets remainder
                end = n
            else:
                end = (k + 1) * fold_size
            folds.append(indices[start:end])

        return folds

    def split_three_way(
        self, train_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split training indices for 3-way cross-fitting.

        Args:
            train_indices: Indices not in test fold

        Returns:
            (theta_indices, lambda_indices)
        """
        n_train = len(train_indices)
        n_theta = int(n_train * self.three_way_theta_frac)

        # Shuffle before splitting
        shuffled = train_indices.copy()
        np.random.shuffle(shuffled)

        theta_indices = shuffled[:n_theta]
        lambda_indices = shuffled[n_theta:]

        return theta_indices, lambda_indices


def run_crossfit(
    Y: Tensor,
    T: Tensor,
    X: Tensor,
    t_tilde: Tensor,
    model: "StructuralModel",
    target: "Target",
    lambda_strategy: "LambdaStrategy",
    n_folds: int = 50,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    hidden_dims: List[int] = [64, 32],
    tikhonov_scale: float = 0.01,
    variance: str = "pooled",
    verbose: bool = False,
    network_factory: Optional["Callable"] = None,
    n_repeats: int = 1,
    three_way_theta_frac: float = 0.6,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    base_seed: Optional[int] = None,
) -> CrossFitResult:
    """
    Run cross-fitting procedure (optionally repeated with median aggregation).

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        t_tilde: Evaluation point
        model: Structural model
        target: Target functional
        lambda_strategy: Lambda estimation strategy
        n_folds: Number of folds
        epochs: Training epochs per fold
        lr: Learning rate
        hidden_dims: Hidden layer sizes
        tikhonov_scale: Scale ε for the Tikhonov Lambda inversion
            (Λ + εI)⁻¹ with ε = tikhonov_scale·trace(Λ)/d (default 0.01).
        variance: Variance estimator, 'pooled' (default, FLM) or 'within_fold'
            (legacy per-fold-centered). See engine.variance.compute_se_ci.
        verbose: Print progress
        n_repeats: Number of repeated cross-fitting splits (default 1 == the
            single-split path; reduces EXACTLY to it). For n_repeats > 1 the
            point estimate and SE are aggregated across splits with the
            Chernozhukov et al. (2018) median DML rule:
                mu_hat = median_r(mu_r)
                se^2   = median_r( se_r^2 + (mu_r - mu_hat)^2 )
            The (mu_r - mu_hat)^2 term folds the across-split variation INTO the
            SE (it widens, never averages it away). CI = mu_hat ± z·se.
        three_way_theta_frac: Fraction of the training data used to fit theta in
            the 3-way (Regime C) split (default 0.6 == previous hardcoded value).
            Lowering it shrinks the theta sample (an undersmoothing knob).
        dropout: Dropout probability for the StructuralNet hidden layers
            (default 0.1 == previous hardcoded value). Ignored when a
            network_factory is supplied (the factory controls architecture).
        weight_decay: L2 penalty for the Adam optimizer (default 0.0 == no
            penalty, identical to the previous optimizer).
        base_seed: Base RNG seed for the per-repeat CrossFitter. When None and
            n_repeats == 1 the single CrossFitter uses random_state=None (the
            exact previous behavior). Otherwise repeat r uses random_state =
            (base_seed or 0) + r, giving reproducible distinct splits.

    Returns:
        CrossFitResult with all inference outputs. For n_repeats > 1 the stored
        psi_values / theta_hat / fold_results are from the FIRST repeat; only the
        scalar mu_hat / se / CI are median-aggregated.
    """
    from deep_inference.models import StructuralNet, train_structural_net
    from deep_inference.engine.assembler import compute_psi
    from deep_inference.engine.variance import (
        compute_inference_results,
        median_dml_aggregate,
    )

    n = Y.shape[0]
    d_x = X.shape[1]
    d_theta = model.theta_dim
    device = Y.device
    dtype = Y.dtype

    # Determine if 3-way split needed
    three_way = lambda_strategy.requires_separate_fold

    def _run_single_split(random_state):
        """One full cross-fitting pass. Behaviorally identical to the original
        single-split body when random_state=None. Returns
        (all_psi, all_theta, fold_results, fold_indices)."""
        # Create cross-fitter
        crossfitter = CrossFitter(
            n_folds=n_folds,
            three_way_theta_frac=three_way_theta_frac,
            random_state=random_state,
        )
        folds = crossfitter.create_folds(n)

        # Storage for all observations
        all_psi = torch.zeros(n, dtype=dtype, device=device)
        all_theta = torch.zeros(n, d_theta, dtype=dtype, device=device)
        fold_results = []

        # Cross-fitting loop
        fold_iterator = list(enumerate(folds))
        if verbose and HAS_TQDM:
            fold_iterator = tqdm(fold_iterator, desc="Cross-fitting", ncols=80)

        for k, eval_fold in fold_iterator:
            if verbose and not HAS_TQDM:
                print(f"Fold {k+1}/{n_folds}")

            # Get training indices (everything except eval fold)
            train_mask = np.ones(n, dtype=bool)
            train_mask[eval_fold] = False
            train_indices = np.where(train_mask)[0]

            if three_way:
                # 3-way split: separate theta and lambda training
                theta_indices, lambda_indices = crossfitter.split_three_way(train_indices)
            else:
                # 2-way split: use all training data for both
                theta_indices = train_indices
                lambda_indices = train_indices

            # Get data subsets
            X_theta = X[theta_indices]
            T_theta = T[theta_indices]
            Y_theta = Y[theta_indices]

            X_lambda = X[lambda_indices]
            T_lambda = T[lambda_indices]
            Y_lambda = Y[lambda_indices]

            X_eval = X[eval_fold]
            T_eval = T[eval_fold]
            Y_eval = Y[eval_fold]

            # 1. Train theta network
            if network_factory is not None:
                theta_net = network_factory(d_x, d_theta)
            else:
                theta_net = StructuralNet(
                    input_dim=d_x,
                    theta_dim=d_theta,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )

            def loss_fn_batched(y, t, theta):
                """Batched loss for training."""
                losses = torch.zeros(len(y), dtype=dtype, device=device)
                for i in range(len(y)):
                    losses[i] = model.loss(y[i], t[i], theta[i])
                return losses

            history = train_structural_net(
                model=theta_net,
                X=X_theta,
                T=T_theta,
                Y=Y_theta,
                loss_fn=loss_fn_batched,
                epochs=epochs,
                lr=lr,
                patience=patience,
                weight_decay=weight_decay,
                verbose=False,
            )

            # Get theta predictions
            with torch.no_grad():
                theta_hat_lambda = theta_net(X_lambda)
                theta_hat_eval = theta_net(X_eval)

            # 2. Fit/compute Lambda
            if lambda_strategy.requires_theta:
                lambda_strategy.fit(
                    X=X_lambda,
                    T=T_lambda,
                    Y=Y_lambda,
                    theta_hat=theta_hat_lambda,
                    model=model,
                )
            else:
                lambda_strategy.fit(
                    X=X_lambda,
                    T=T_lambda,
                    Y=Y_lambda,
                    theta_hat=None,
                    model=model,
                )

            # 3. Get Lambda for eval observations
            lambda_eval = lambda_strategy.predict(X_eval, theta_hat_eval)

            # 4. Compute psi on eval fold
            psi_eval = compute_psi(
                Y=Y_eval,
                T=T_eval,
                X=X_eval,
                theta_hat=theta_hat_eval,
                t_tilde=t_tilde,
                lambda_matrices=lambda_eval,
                model=model,
                target=target,
                tikhonov_scale=tikhonov_scale,
            )

            # Store results
            all_psi[eval_fold] = psi_eval
            all_theta[eval_fold] = theta_hat_eval

            fold_results.append(
                FoldResult(
                    fold_idx=k,
                    psi_values=psi_eval,
                    theta_hat=theta_hat_eval,
                    eval_indices=list(eval_fold),
                    train_history={"val_loss": history.val_losses[-1]} if history.val_losses else None,
                )
            )

        # Build a (n,) fold-assignment vector for the within_fold variant.
        fold_indices = np.zeros(n, dtype=np.int64)
        for k, eval_fold in enumerate(folds):
            fold_indices[eval_fold] = k

        return all_psi, all_theta, fold_results, fold_indices

    # Repeated cross-fitting with median DML aggregation. n_repeats == 1 with
    # base_seed=None reproduces the original single CrossFitter (random_state=None)
    # exactly, and the branch below returns its compute_inference_results verbatim.
    repeat_results = []  # per-repeat dicts from compute_inference_results
    psi_keep = theta_keep = fold_results_keep = None

    for r in range(n_repeats):
        if n_repeats == 1 and base_seed is None:
            random_state = None  # exact original path
        else:
            random_state = (base_seed if base_seed is not None else 0) + r

        all_psi, all_theta, fold_results, fold_indices = _run_single_split(random_state)
        res = compute_inference_results(
            all_psi, method=variance, fold_indices=fold_indices
        )
        repeat_results.append(res)
        if r == 0:
            psi_keep, theta_keep, fold_results_keep = all_psi, all_theta, fold_results

    if n_repeats == 1:
        # Verbatim single-split inference results -> byte-identical to before.
        agg = repeat_results[0]
        mu_hat = agg["mu_hat"]
        se = agg["se"]
        ci_lower = agg["ci_lower"]
        ci_upper = agg["ci_upper"]
    else:
        # Chernozhukov et al. (2018) median DML rule (single source of truth).
        mu_hat, se, ci_lower, ci_upper = median_dml_aggregate(
            [res["mu_hat"] for res in repeat_results],
            [res["se"] for res in repeat_results],
            alpha=0.05,
        )

    return CrossFitResult(
        psi_values=psi_keep,
        theta_hat=theta_keep,
        mu_hat=mu_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        fold_results=fold_results_keep,
    )
