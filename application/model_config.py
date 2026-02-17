#!/usr/bin/env python3
"""
model_config.py - Model configuration for choice models.

Copied from deep-aesthetics project for reference.
Original: /Users/pranjal/Dropbox/deep-aesthetics/final-analysis/choice/model_config.py
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration for MNL choice model training"""

    # Model architecture
    model_type: Literal["two_tower", "mlp", "linear"] = "two_tower"
    K_rank: int = 256
    use_dropout: bool = True
    dropout_rate: float = 0.2

    # Embedding source
    use_fused_embeddings: bool = False
    normalize_product_emb: bool = True

    # Alpha (price sensitivity) model
    alpha_architecture: Literal["mlp", "linear"] = "mlp"
    alpha_hidden_dim: int = 32

    # Training
    epochs: int = 20
    lr_alpha: float = 5e-5
    lr_beta: float = 1e-4
    weight_decay_alpha: float = 1e-3
    weight_decay_beta: float = 3e-3
    use_scheduler: bool = True
    scheduler_min_lr_ratio: float = 0.1
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 10

    # Data
    choice_set_mode: Literal["ragged", "fixed"] = "ragged"

    # Logging
    log_every: int = 5
    log_gradients: bool = True

@dataclass
class ClusteredModelConfig(ModelConfig):
    """Configuration for cluster-aggregated MNL training"""
    n_clusters: int = 500
    cluster_batch_size: int = 2048
    patience: int = 15
    epochs: int = 100

@dataclass
class IncidenceModelConfig:
    """Configuration for incidence (purchase/no-purchase) model"""
    epochs: int = 15
    lr: float = 0.01
    use_recency: bool = True
    use_seasonality: bool = True


# Preset configurations
CONFIGS = {
    "baseline": ModelConfig(
        model_type="two_tower",
        K_rank=256,  # Increased from 128 to preserve more signal
        epochs=20,
        lr_alpha=1e-4,
        lr_beta=1e-4,
        weight_decay_beta=1e-4,  # Reduced from 3e-3 to allow projections to grow
        use_dropout=False,
        normalize_product_emb=False,  # Disabled to preserve variance
        choice_set_mode="fixed"
    ),

    "ragged_100ep": ModelConfig(
        model_type="two_tower",
        K_rank=256,
        epochs=100,
        lr_alpha=5e-5,
        lr_beta=1e-4,
        use_dropout=True,
        dropout_rate=0.2,
        choice_set_mode="ragged",
        normalize_product_emb=True
    ),

    "linear_baseline": ModelConfig(
        model_type="linear",
        K_rank=64,
        epochs=30,
        lr_alpha=1e-3,
        lr_beta=1e-3,
        use_dropout=False,
        alpha_architecture="linear"
    ),

    "mlp_deep": ModelConfig(
        model_type="mlp",
        K_rank=128,
        epochs=50,
        lr_alpha=5e-5,
        lr_beta=5e-5,
        use_dropout=True,
        dropout_rate=0.3
    ),

    "clustered_500": ClusteredModelConfig(
        model_type="two_tower",
        K_rank=256,
        n_clusters=500,
        epochs=100,
        lr_alpha=1e-4,
        lr_beta=2e-4
    ),

    "test_scaled": ModelConfig(
        model_type="two_tower",
        K_rank=128,
        epochs=5,
        lr_alpha=1e-4,
        lr_beta=1e-4,
        use_dropout=False,
        normalize_product_emb=False,
        choice_set_mode="ragged"
    ),

    "small_sample": ModelConfig(
        model_type="two_tower",
        K_rank=32,
        epochs=200,
        lr_alpha=1e-3,
        lr_beta=2e-3,
        use_dropout=True,
        dropout_rate=0.1,
        normalize_product_emb=False,
        choice_set_mode="ragged",
        patience=20,
        weight_decay_alpha=1e-4,
        weight_decay_beta=1e-4,
        use_scheduler=True,
        scheduler_min_lr_ratio=0.01,
    ),

    "full_sample": ModelConfig(
        model_type="two_tower",
        K_rank=64,
        epochs=100,
        lr_alpha=5e-4,
        lr_beta=1e-3,
        use_dropout=True,
        dropout_rate=0.15,
        normalize_product_emb=False,
        choice_set_mode="ragged",
        patience=15,
        weight_decay_alpha=1e-4,
        weight_decay_beta=1e-4,
        use_scheduler=True,
        scheduler_min_lr_ratio=0.01,
    ),
}


def get_config(name: str = "ragged_100ep") -> ModelConfig:
    """Get config by name, defaults to current production config"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
