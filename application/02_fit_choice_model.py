#!/usr/bin/env python3
"""
02_fit_choice_model.py: Vectorized Two-Tower MNL per Spec

Copied from deep-aesthetics project for reference.
Original: /Users/pranjal/Dropbox/deep-aesthetics/final-analysis/choice/04_fit_two_tower.py

Model: U_ijt = alpha(d_i)*p_jt + <r_i, s_j> + b
- alpha(d): Price sensitivity via softplus MLP (always negative, units: per EUR)
- r_i = P_d(d_i): Customer projection [I, K]
- s_j = P_x(x_j): Product projection [J, K]
- b: Scalar bias

Training: Vectorized weekly softmax (no I*J materialization)
Outputs: artifacts/mnl/two_tower_{config}.pt
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
# Original used: from config import DATA_PATH, DATA_DIR
# and: import choice_config as ccfg
# Adapt to deepest application layout.
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data"
DATA_DIR = APP_DIR / "data"

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime
from model_config import get_config

# ── Output/Artifact directories ────────────────────────────────────────
# Original used choice_config (ccfg) for these; adapt to local layout.
EMB_VARIANT = os.environ.get("EMB_VARIANT", "default")
OUTPUT_DIR = APP_DIR / "outputs"
ARTIFACTS_DIR = APP_DIR / "artifacts"

OUTPUT_DIR.mkdir(exist_ok=True)
(ARTIFACTS_DIR / "outputs").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "mnl").mkdir(parents=True, exist_ok=True)

# Load config
CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else "small_sample"
config = get_config(CONFIG_NAME)

sys.stdout = open(str(ARTIFACTS_DIR / "outputs" / f"fit_two_tower_{CONFIG_NAME}.txt"), 'w', buffering=1)

print("=" * 80)
print("FIT_TWO_TOWER: Vectorized Two-Tower MNL")
print("=" * 80)
print(f"Config: {CONFIG_NAME}, EMB_VARIANT={EMB_VARIANT}")
print(f"  K={config.K_rank}, dropout={config.use_dropout}")
print(f"  Epochs: {config.epochs}, LR_alpha={config.lr_alpha}, LR_beta={config.lr_beta}")
print(f"  Artifacts: {ARTIFACTS_DIR}")
print(f"Start: {datetime.now()}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
# Route to prep_full/ when using full_sample config
PREP_SUBDIR = "prep_full" if CONFIG_NAME == "full_sample" else "prep"
PREP_DIR = ARTIFACTS_DIR / PREP_SUBDIR

print(f"Loading artifacts from {PREP_DIR}...")
cust = pl.read_parquet(str(PREP_DIR / "customers.parquet"))
art = pl.read_parquet(str(PREP_DIR / "articles.parquet"))
txn_train = pl.read_parquet(str(PREP_DIR / "purchases_train.parquet"))
txn_val = pl.read_parquet(str(PREP_DIR / "purchases_val.parquet"))

price_data = np.load(str(PREP_DIR / "price_jt.npz"))
p_jt = price_data['p']  # [J, T], NaN = unavailable
avail_jt = price_data['avail']  # [J, T], 1 = available

with open(str(PREP_DIR / "maps.pkl"), "rb") as f:
    maps = pickle.load(f)

I, J, T = maps['I'], maps['J'], maps['T']
cust_to_i = maps['cust_to_i']
sku_to_j = maps['sku_to_j']
week_to_t = maps['week_to_t']

print(f"Loaded: I={I:,} customers, J={J} SKUs, T={T} weeks")
print(f"Train: {len(txn_train):,}, Val: {len(txn_val):,}\n")

# Load embeddings
d_matrix = np.load(str(PREP_DIR / "d_matrix.npy"))
x_matrix = np.load(str(PREP_DIR / "x_matrix.npy"))

d_tensor = torch.tensor(d_matrix, dtype=torch.float32)
x_tensor = torch.tensor(x_matrix, dtype=torch.float32)

# NO normalization - preserve natural embedding variance for product differentiation
# L2 normalization collapses all products to unit sphere, killing discriminative power
print("Skipping L2 normalization to preserve product variance")

# Linear prices (EUR) -- elasticity eps = alpha*p*(1-s) becomes price-dependent
p_jt_tensor = torch.tensor(p_jt, dtype=torch.float32)
p_jt_tensor = torch.nan_to_num(p_jt_tensor, nan=0.0)
p_min, p_max = np.nanmin(p_jt), np.nanmax(p_jt)
print(f"Price: EUR{p_min:.2f} - EUR{p_max:.2f}, linear (EUR)")

avail_jt_tensor = torch.tensor(avail_jt, dtype=torch.bool)

print(f"Embeddings: d={d_tensor.shape}, x={x_tensor.shape}")
print(f"Price: {p_jt_tensor.shape}, Availability: {avail_jt_tensor.shape}\n")

# ============================================================================
# PRECOMPUTE WEEKLY AVAILABILITY
# ============================================================================
print("Precomputing weekly availability...")
avail_idx_by_t = [torch.nonzero(avail_jt_tensor[:, t]).squeeze(1) for t in range(T)]
avg_avail = np.mean([len(idx) for idx in avail_idx_by_t])
print(f"Average available SKUs per week: {avg_avail:.0f} / {J} ({100*avg_avail/J:.1f}%)\n")

# ============================================================================
# GROUP PURCHASES BY WEEK
# ============================================================================
def group_by_week(txn_df, split_name):
    """Group purchases by week, map chosen j to position in that week's available set"""
    print(f"Grouping {split_name} occasions by week...")

    occasions = []
    for row in txn_df.iter_rows(named=True):
        i = cust_to_i.get(row["customer_id_idx"])
        j = sku_to_j.get(row["article_id_idx"])
        t = week_to_t.get(row["week_start"])
        if i is not None and j is not None and t is not None:
            occasions.append({"i": i, "j": j, "t": t})

    # Group by week
    by_t = {t: [] for t in range(T)}
    for occ in occasions:
        by_t[occ['t']].append((occ['i'], occ['j']))

    # Convert to tensors with position mapping
    i_by_t, pos_by_t = {}, {}
    total_occasions = 0

    for t in range(T):
        j_idx = avail_idx_by_t[t]
        j_to_pos = {j.item(): pos for pos, j in enumerate(j_idx)}
        occs = by_t[t]

        if len(occs) == 0:
            continue

        # Filter occasions where chosen j is available this week
        valid_occs = [(i, j) for i, j in occs if j in j_to_pos]

        if len(valid_occs) == 0:
            continue

        i_by_t[t] = torch.tensor([o[0] for o in valid_occs], dtype=torch.long)
        pos_by_t[t] = torch.tensor([j_to_pos[o[1]] for o in valid_occs], dtype=torch.long)
        total_occasions += len(valid_occs)

    print(f"  {split_name}: {total_occasions:,} occasions across {len(i_by_t)} weeks")
    return i_by_t, pos_by_t, total_occasions

train_i_by_t, train_pos_by_t, n_train = group_by_week(txn_train, "TRAIN")
val_i_by_t, val_pos_by_t, n_val = group_by_week(txn_val, "VAL")
print()

# ============================================================================
# MODELS
# ============================================================================
print("Initializing models...")

class AlphaMLP(nn.Module):
    """Price sensitivity: alpha(d) = -softplus(h_phi(d))"""
    def __init__(self, d_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, d):
        """d: [batch, D] -> alpha: [batch, 1]"""
        return -torch.nn.functional.softplus(self.net(d))

class TwoTower(nn.Module):
    """Two-tower: g(x,d) = <P_x(x), P_d(d)> + b"""
    def __init__(self, x_dim, d_dim, K, use_dropout=True, dropout_rate=0.2):
        super().__init__()
        self.P_x = nn.Linear(x_dim, K, bias=False)
        if use_dropout:
            self.P_d = nn.Sequential(
                nn.Linear(d_dim, K, bias=False),
                nn.Dropout(dropout_rate)
            )
        else:
            self.P_d = nn.Linear(d_dim, K, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, d):
        """
        x: [J, X] -> s_j: [J, K]
        d: [I, D] -> r_i: [I, K]
        Returns: (s_j, r_i)
        """
        s_j = self.P_x(x)  # [J, K]
        r_i = self.P_d(d)  # [I, K]
        return s_j, r_i

alpha_model = AlphaMLP(
    d_dim=d_tensor.shape[1],
    hidden_dim=config.alpha_hidden_dim
)
beta_model = TwoTower(
    x_dim=x_tensor.shape[1],
    d_dim=d_tensor.shape[1],
    K=config.K_rank,
    use_dropout=config.use_dropout,
    dropout_rate=config.dropout_rate
)

total_params = sum(p.numel() for p in alpha_model.parameters()) + sum(p.numel() for p in beta_model.parameters())
print(f"Model: alpha(MLP, hidden={config.alpha_hidden_dim}), beta(two-tower, K={config.K_rank})")
print(f"Parameters: {total_params:,}\n")

# ============================================================================
# OPTIMIZERS
# ============================================================================
optimizer_alpha = optim.AdamW(
    alpha_model.parameters(),
    lr=config.lr_alpha,
    weight_decay=config.weight_decay_alpha
)
optimizer_beta = optim.AdamW(
    beta_model.parameters(),
    lr=config.lr_beta,
    weight_decay=config.weight_decay_beta
)

if config.use_scheduler:
    scheduler_alpha = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_alpha,
        T_max=config.epochs,
        eta_min=config.lr_alpha * config.scheduler_min_lr_ratio
    )
    scheduler_beta = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_beta,
        T_max=config.epochs,
        eta_min=config.lr_beta * config.scheduler_min_lr_ratio
    )
else:
    scheduler_alpha = scheduler_beta = None

print(f"Training {config.epochs} epochs (vectorized weekly softmax)...\n")

# ============================================================================
# TRAINING LOOP
# ============================================================================
best_val_nll = float('inf')
best_epoch = 0
patience_counter = 0

for epoch in range(config.epochs):
    # === TRAIN ===
    alpha_model.train()
    beta_model.train()
    optimizer_alpha.zero_grad()
    optimizer_beta.zero_grad()

    # Compute projections once per epoch
    s_j_all, r_i_all = beta_model(x_tensor, d_tensor)  # [J, K], [I, K]
    alpha_all = alpha_model(d_tensor)  # [I, 1]
    b = beta_model.b

    # Aggregate loss across weeks
    total_loss = 0.0
    total_occasions = 0

    for t in train_i_by_t.keys():
        j_idx = avail_idx_by_t[t]  # available SKUs this week [J_t]
        i_batch = train_i_by_t[t]  # customers purchasing this week [n_t]
        pos = train_pos_by_t[t]    # chosen j's position in j_idx [n_t]

        if len(i_batch) == 0:
            continue

        # g component: r_i @ s_j.T + b
        # [n_t, K] @ [K, J_t] = [n_t, J_t]
        g_week = r_i_all[i_batch] @ s_j_all[j_idx].T + b

        # alpha*p component
        alpha_week = alpha_all[i_batch]  # [n_t, 1]
        p_week = p_jt_tensor[j_idx, t].unsqueeze(0)  # [1, J_t]

        # Utility: U_ijt = alpha_i*p_jt + <r_i, s_j> + b
        V_week = alpha_week * p_week + g_week  # [n_t, J_t]

        # Cross-entropy loss for this week
        loss_week = torch.nn.functional.cross_entropy(V_week, pos, reduction='sum')
        total_loss += loss_week
        total_occasions += len(i_batch)

    loss_train = total_loss / total_occasions

    # Debug first epoch
    if epoch == 0:
        print(f"[Epoch 0 debug] alpha range: [{alpha_all.min().item():.4f}, {alpha_all.max().item():.4f}]")
        print(f"[Epoch 0 debug] s_j range: [{s_j_all.min().item():.4f}, {s_j_all.max().item():.4f}]")
        print(f"[Epoch 0 debug] r_i range: [{r_i_all.min().item():.4f}, {r_i_all.max().item():.4f}]")
        print(f"[Epoch 0 debug] b: {b.item():.4f}")
        print(f"[Epoch 0 debug] train loss: {loss_train.item():.4f}\n")

    loss_train.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), config.grad_clip)
    torch.nn.utils.clip_grad_norm_(beta_model.parameters(), config.grad_clip)

    optimizer_alpha.step()
    optimizer_beta.step()

    # === VALIDATION ===
    if (epoch + 1) % config.log_every == 0 or epoch == 0:
        alpha_model.eval()
        beta_model.eval()

        with torch.no_grad():
            # Recompute projections in eval mode
            s_j_eval, r_i_eval = beta_model(x_tensor, d_tensor)
            alpha_eval = alpha_model(d_tensor)
            b_eval = beta_model.b

            # TRAIN metrics
            train_correct = 0
            train_top5_correct = 0
            for t in train_i_by_t.keys():
                j_idx = avail_idx_by_t[t]
                i_batch = train_i_by_t[t]
                pos = train_pos_by_t[t]
                if len(i_batch) == 0:
                    continue

                g_week = r_i_eval[i_batch] @ s_j_eval[j_idx].T + b_eval
                alpha_week = alpha_eval[i_batch]
                p_week = p_jt_tensor[j_idx, t].unsqueeze(0)
                V_week = alpha_week * p_week + g_week

                preds = V_week.argmax(dim=1)
                train_correct += (preds == pos).sum().item()

                top5 = torch.topk(V_week, k=min(5, len(j_idx)), dim=1).indices
                train_top5_correct += sum((pos[i] in top5[i]) for i in range(len(pos)))

            acc_train = train_correct / n_train
            top5_acc_train = train_top5_correct / n_train

            # VAL metrics
            val_total_loss = 0.0
            val_correct = 0
            val_top5_correct = 0
            val_occasions = 0

            for t in val_i_by_t.keys():
                j_idx = avail_idx_by_t[t]
                i_batch = val_i_by_t[t]
                pos = val_pos_by_t[t]
                if len(i_batch) == 0:
                    continue

                g_week = r_i_eval[i_batch] @ s_j_eval[j_idx].T + b_eval
                alpha_week = alpha_eval[i_batch]
                p_week = p_jt_tensor[j_idx, t].unsqueeze(0)
                V_week = alpha_week * p_week + g_week

                loss_week = torch.nn.functional.cross_entropy(V_week, pos, reduction='sum')
                val_total_loss += loss_week
                val_occasions += len(i_batch)

                preds = V_week.argmax(dim=1)
                val_correct += (preds == pos).sum().item()

                top5 = torch.topk(V_week, k=min(5, len(j_idx)), dim=1).indices
                val_top5_correct += sum((pos[i] in top5[i]) for i in range(len(pos)))

            loss_val = val_total_loss / val_occasions
            acc_val = val_correct / val_occasions
            top5_acc_val = val_top5_correct / val_occasions

            # Track best model + early stopping
            if loss_val.item() < best_val_nll:
                best_val_nll = loss_val.item()
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(alpha_model.state_dict(), str(ARTIFACTS_DIR / "mnl" / f"alpha_best_two_tower_{CONFIG_NAME}.pt"))
                torch.save(beta_model.state_dict(), str(ARTIFACTS_DIR / "mnl" / f"beta_best_two_tower_{CONFIG_NAME}.pt"))
            else:
                patience_counter += 1

        epoch_marker = '*' if epoch+1==best_epoch else ''
        early_stop_marker = f" [patience: {patience_counter}/{config.patience}]" if patience_counter > 0 else ""
        lr_str = f"LR_alpha={scheduler_alpha.get_last_lr()[0]:.2e}, LR_beta={scheduler_beta.get_last_lr()[0]:.2e}" if config.use_scheduler else ""
        print(f"Epoch {epoch+1:2d}: Train NLL={loss_train.item():.4f}, Top-1={acc_train:.3f}, Top-5={top5_acc_train:.3f} | Val NLL={loss_val.item():.4f}, Top-1={acc_val:.3f}, Top-5={top5_acc_val:.3f} | {lr_str} {epoch_marker}{early_stop_marker}")

        # Early stopping check
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} (patience={config.patience})")
            break

    # Step LR schedulers
    if config.use_scheduler:
        scheduler_alpha.step()
        scheduler_beta.step()

print(f"\nBest model: Epoch {best_epoch}, Val NLL={best_val_nll:.4f}\n")

# ============================================================================
# SAVE MODELS
# ============================================================================
print(f"Saving to {ARTIFACTS_DIR / 'mnl'}...")
torch.save(alpha_model.state_dict(), str(ARTIFACTS_DIR / "mnl" / f"alpha_two_tower_{CONFIG_NAME}.pt"))
torch.save(beta_model.state_dict(), str(ARTIFACTS_DIR / "mnl" / f"beta_two_tower_{CONFIG_NAME}.pt"))

# Save frozen embeddings for fast CV computation
with torch.no_grad():
    s_j, r_i = beta_model(x_tensor, d_tensor)
    s_j = s_j.numpy()
    r_i = r_i.numpy()
    b_val = beta_model.b.item()
    alpha_i = alpha_model(d_tensor).squeeze().numpy()

np.save(str(ARTIFACTS_DIR / "mnl" / f"s_j_two_tower_{CONFIG_NAME}.npy"), s_j)
np.save(str(ARTIFACTS_DIR / "mnl" / f"r_i_two_tower_{CONFIG_NAME}.npy"), r_i)
np.save(str(ARTIFACTS_DIR / "mnl" / f"b_two_tower_{CONFIG_NAME}.npy"), np.array([b_val]))
np.save(str(ARTIFACTS_DIR / "mnl" / f"alpha_i_two_tower_{CONFIG_NAME}.npy"), alpha_i)

# Save config for reproducibility
import json
config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in vars(config).items()}
with open(str(ARTIFACTS_DIR / "mnl" / f"config_two_tower_{CONFIG_NAME}.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"Saved: alpha/beta_two_tower_{CONFIG_NAME}.pt")
print(f"Saved: s_j/r_i/b/alpha_i_two_tower_{CONFIG_NAME}.npy")
print(f"Saved: config_two_tower_{CONFIG_NAME}.json\n")

print("=" * 80)
print("TWO-TOWER MNL TRAINING COMPLETE")
print("=" * 80)
print(f"Config: {CONFIG_NAME}")
print(f"End: {datetime.now()}")
print(f"\nModel: U_ijt = alpha(d_i)*p_jt + <r_i, s_j> + b")
print(f"Final: Train NLL={loss_train.item():.4f}, Val NLL={best_val_nll:.4f}")
print(f"Best epoch: {best_epoch}/{config.epochs}")
print(f"\nNote: Using linear EUR prices -- eps = alpha*p*(1-s) varies with price level")

sys.stdout.close()
