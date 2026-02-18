"""
Configuration for the H&M fashion demand application.

This module defines paths, hyperparameters, and model settings for the
practitioner's guide worked example.
"""

from pathlib import Path

# === Paths ===
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts_arm_a"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT.parent / "paper" / "practitioners_guide" / "figures"

# Deep-aesthetics data (source)
DEEP_AESTHETICS_DIR = Path("/Users/pranjal/Dropbox/deep-aesthetics/final-analysis")
EMBEDDINGS_DIR = DEEP_AESTHETICS_DIR / "embeddings" / "artifacts_arm_a"
TRANSACTIONS_DIR = DEEP_AESTHETICS_DIR / "choice"
CHOICE_PREP_DIR = DEEP_AESTHETICS_DIR / "choice" / "artifacts_arm_a" / "prep"

# === Data Settings ===
# Choice set construction
J = 20              # Number of alternatives per choice occasion (1 chosen + 19 sampled)
K = 6               # Number of attributes per alternative (log_price + 5 PCA dims)
PCA_DIMS = 5        # PCA dimensions for item embeddings (64D -> 5D)
MIN_PURCHASES = 5   # Minimum purchases per consumer

# === Model Settings ===
THETA_DIM = K       # Heterogeneous parameters: [beta_price, beta_pca1, ..., beta_pca5]

# === Neural Network Settings ===
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 0.01
EPOCHS = 300
PATIENCE = 50       # CRITICAL: must be >= 50 for multinomial with 3-way split
N_FOLDS = 50
DROPOUT = 0.1

# === Lambda Settings ===
LAMBDA_METHOD = "ridge"
RIDGE_ALPHA = 1000.0

# === Simulation Settings ===
SIM_N = 8000        # Sample size for MC validation (n=5000 underfits with 3-way split)
SIM_M = 50          # Number of MC replications
SIM_D_X = 10        # Consumer embedding dimension (reduced for speed)
SIM_SEED = 42

# === Attribute Names ===
ATTRIBUTE_NAMES = ["log_price", "pca_1", "pca_2", "pca_3", "pca_4", "pca_5"]
