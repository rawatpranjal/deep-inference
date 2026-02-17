#!/usr/bin/env python3
"""
01_train_embeddings.py: Arm A -- InfoNCE + Price in Item Tower

Copied from deep-aesthetics project for reference.
Original: /Users/pranjal/Dropbox/deep-aesthetics/final-analysis/embeddings/08_train_infonce_v2.py

A/B Test Arm A: InfoNCE-only contrastive learning with price fed through
the item tower. Compared against Arm B (joint choice model) to isolate
the effect of MNL loss on downstream analyses.

ItemTowerV2: CLIP(512) + Cat(26) + log(ref_price)(1) = 539D input
UserTowerV2: ID_emb(64) + Cat(6) + Cont(7) = 77D input

Artifacts (artifacts_arm_a/):
  user_embeddings.npy, item_embeddings.npy
  id_maps.pkl, model_state.pt, clip_features.npy
  checkpoint.pt, prep_cache.npz
  ref_prices.npy         median price per T1 item
  ref_prices_all.npy     median price per ALL dress items (for lockdown extension)
  tower_config.json      {has_price_feature: true, input_dim: 539, ...}
"""

import sys
import time
import json
import pickle
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
# Original used: from config import DATA_DIR, CLIP_PARQUET
# Adapt to deepest application layout. Set these before running.
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
V2_DIR = APP_DIR / "artifacts_infonce_v2"
ARTIFACTS_DIR = APP_DIR / "artifacts_arm_a"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

log_file = open(OUTPUT_DIR / '01_train_embeddings.txt', 'w', buffering=1)


def log(msg=""):
    """Print to both log file and console."""
    log_file.write(msg + "\n")
    log_file.flush()
    print(msg, file=sys.stderr)


# ── Hyperparameters ─────────────────────────────────────────────────
K = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
BATCH_SIZE = 1024
TAU = 0.1
DROPOUT = 0.1
GRAD_CLIP = 1.0
PATIENCE = 10
VAL_FRAC = 0.1
SEED = 42

T1_START = datetime(2018, 9, 1)
T1_END = datetime(2019, 7, 1)  # exclusive
# Behavioral features: use STORE-channel T1 transactions (training uses ONLINE only)
# H&M dataset starts 2018-09-20, so no pre-T1 data exists. Store purchases during
# T1 are non-overlapping with online dress interactions -> zero leakage.
BEH_START = datetime(2018, 9, 1)
BEH_END = datetime(2019, 7, 1)
STORE_CHANNEL = 1  # behavioral features from store; training uses online (2)

# Item categorical config: (column_name, embedding_dim)
ITEM_CAT_COLS = [
    'colour_group_name',           # ~50 values, dim 8
    'graphical_appearance_name',   # ~30 values, dim 4
    'perceived_colour_value_name', # ~8 values, dim 3
    'perceived_colour_master_name',# ~20 values, dim 4
    'section_name',                # ~56 values, dim 4
    'index_name',                  # ~10 values, dim 3
]
ITEM_CAT_DIMS = [8, 4, 3, 4, 4, 3]  # 26 total

# User categorical config
USER_CAT_COLS = ['club_member_status', 'fashion_news_frequency']
USER_CAT_DIMS = [3, 3]  # 6 total
N_USER_CONT = 7  # age, FN, Active, log_mean_price, log_n_types, log_n_purch, dress_frac

torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

log("=" * 80)
log("01_TRAIN_EMBEDDINGS: Arm A -- InfoNCE + Price in Item Tower")
log("=" * 80)
log(f"A/B test: Arm A (InfoNCE + price) vs Arm B (joint choice)")
log(f"K={K}, lr={LR}, wd={WEIGHT_DECAY}, epochs={EPOCHS}, batch={BATCH_SIZE}")
log(f"tau={TAU}, dropout={DROPOUT}, patience={PATIENCE}, val_frac={VAL_FRAC}")
log(f"T1: {T1_START.date()} -> {T1_END.date()} (training window, online)")
log(f"Behavioral: {BEH_START.date()} -> {BEH_END.date()} (store channel, all products)")
log(f"Item categoricals: {ITEM_CAT_COLS}")
log(f"  dims: {ITEM_CAT_DIMS} = {sum(ITEM_CAT_DIMS)}d total")
log(f"User categoricals: {USER_CAT_COLS}, dims: {USER_CAT_DIMS} = {sum(USER_CAT_DIMS)}d")
log(f"User continuous: {N_USER_CONT}d")
log(f"Device: {device}")
log(f"Start: {datetime.now()}\n")

t0 = time.time()

# ════════════════════════════════════════════════════════════════════
# DATA PREP — Reuse v2 prep cache (same data, same split)
# ════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = ARTIFACTS_DIR / "checkpoint.pt"
V2_PREP_CACHE = V2_DIR / "prep_cache.npz"
V2_ID_MAPS = V2_DIR / "id_maps.pkl"
V2_CLIP_FEAT = V2_DIR / "clip_features.npy"

if V2_PREP_CACHE.exists() and V2_ID_MAPS.exists() and V2_CLIP_FEAT.exists():
    log("-- Loading v2 data prep (shared data split) --")
    cache = np.load(str(V2_PREP_CACHE))
    train_pairs = cache["train_pairs"]
    val_pairs = cache["val_pairs"]
    item_cat_features = cache["item_cat_features"]
    user_cat_features = cache["user_cat_features"]
    user_cont_features = cache["user_cont_features"]
    clip_features = np.load(str(V2_CLIP_FEAT))
    with open(str(V2_ID_MAPS), "rb") as f:
        id_maps = pickle.load(f)
    user_to_idx = id_maps["user_to_idx"]
    idx_to_user = id_maps["idx_to_user"]
    item_to_idx = id_maps["item_to_idx"]
    idx_to_item = id_maps["idx_to_item"]
    item_cat_vocabs = id_maps["item_cat_vocabs"]
    user_cat_vocabs = id_maps["user_cat_vocabs"]
    N_users = len(user_to_idx)
    N_items = len(item_to_idx)
    log(f"Loaded: N_users={N_users:,}, N_items={N_items}, "
        f"train={len(train_pairs):,}, val={len(val_pairs):,}")
    log(f"CLIP features: {clip_features.shape}")
    log(f"Item categoricals: {item_cat_features.shape}")
    log(f"User categoricals: {user_cat_features.shape}")
    log(f"User continuous: {user_cont_features.shape}")
    log(f"Item cat vocab sizes: {[len(v) for v in item_cat_vocabs.values()]}")
    log(f"User cat vocab sizes: {[len(v) for v in user_cat_vocabs.values()]}")
else:
    raise FileNotFoundError(
        f"Arm A requires v2 prep artifacts. Run original v2 training first.\n"
        f"Missing: {V2_PREP_CACHE}, {V2_ID_MAPS}, or {V2_CLIP_FEAT}"
    )

log(f"\nData prep: {time.time() - t0:.1f}s")

# ════════════════════════════════════════════════════════════════════
# COMPUTE REFERENCE PRICES (median price per item from T1 transactions)
# ════════════════════════════════════════════════════════════════════
REF_PRICES_PATH = ARTIFACTS_DIR / "ref_prices.npy"
REF_PRICES_ALL_PATH = ARTIFACTS_DIR / "ref_prices_all.npy"

if REF_PRICES_PATH.exists():
    log("\n-- Loading cached ref prices --")
    ref_prices = np.load(str(REF_PRICES_PATH))
    ref_prices_all_dict = {}  # not needed during training
    log(f"ref_prices: {ref_prices.shape}, range=[{ref_prices.min():.6f}, {ref_prices.max():.6f}]")
else:
    log("\n-- Computing reference prices --")
    txn_price = pl.read_csv(
        str(DATA_DIR / "transactions_train.csv"),
        columns=["article_id", "price", "sales_channel_id", "t_dat"],
    ).with_columns(
        pl.col("t_dat").str.to_date("%Y-%m-%d").alias("t_dat")
    )
    # T1 online transactions for items in our set
    t1_prices = txn_price.filter(
        (pl.col("article_id").is_in(list(item_to_idx.keys()))) &
        (pl.col("sales_channel_id") == 2) &
        (pl.col("t_dat") >= T1_START.date()) &
        (pl.col("t_dat") < T1_END.date())
    ).group_by("article_id").agg(
        pl.col("price").median().alias("median_price")
    )

    price_dict = {}
    for row in t1_prices.iter_rows():
        price_dict[row[0]] = row[1]

    # Per-item ref prices (aligned to item_to_idx)
    global_median = float(np.median(list(price_dict.values())))
    ref_prices = np.full(N_items, global_median, dtype=np.float32)
    for aid, j in item_to_idx.items():
        if aid in price_dict:
            ref_prices[j] = price_dict[aid]

    np.save(str(REF_PRICES_PATH), ref_prices)
    log(f"ref_prices: {ref_prices.shape}, range=[{ref_prices.min():.6f}, {ref_prices.max():.6f}]")
    log(f"  global_median={global_median:.6f}")

    # ALL dress items (for lockdown extension of non-T1 items)
    art = pl.read_csv(str(DATA_DIR / "articles.csv"))
    all_dress_ids = set(
        art.filter(pl.col("product_type_name") == "Dress")["article_id"].to_list()
    )
    all_prices = txn_price.filter(
        pl.col("article_id").is_in(list(all_dress_ids))
    ).group_by("article_id").agg(
        pl.col("price").median().alias("median_price")
    )
    ref_prices_all_dict = {}
    for row in all_prices.iter_rows():
        ref_prices_all_dict[row[0]] = row[1]
    np.save(str(REF_PRICES_ALL_PATH), ref_prices_all_dict)
    log(f"ref_prices_all: {len(ref_prices_all_dict):,} dress items")
    del txn_price, art

# Log-transform for tower input
log_ref_prices = np.log(ref_prices + 1e-6).astype(np.float32)
log(f"log_ref_prices: mean={log_ref_prices.mean():.4f}, std={log_ref_prices.std():.4f}")

# ════════════════════════════════════════════════════════════════════
# MODEL — InfoNCE with Price in Item Tower
# ════════════════════════════════════════════════════════════════════
log("\n-- Model --")

# Vocab sizes for embeddings: max index + 1 (0 = unknown)
item_cat_vocab_sizes = [max(v.values()) + 1 for v in item_cat_vocabs.values()]
user_cat_vocab_sizes = [max(v.values()) + 1 for v in user_cat_vocabs.values()]
log(f"Item cat vocab sizes: {item_cat_vocab_sizes}")
log(f"User cat vocab sizes: {user_cat_vocab_sizes}")


class ItemTowerV2(nn.Module):
    """CLIP(512) + Cat(26) + log(ref_price)(1) = 539D -> 64D embedding."""
    def __init__(self, clip_dim, k, cat_vocab_sizes, cat_dims, dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vs, d) for vs, d in zip(cat_vocab_sizes, cat_dims)
        ])
        total_cat_dim = sum(cat_dims)
        in_dim = clip_dim + total_cat_dim + 1  # 512 + 26 + 1(price) = 539

        self.fc1 = nn.Linear(in_dim, 256)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, k)
        self.skip = nn.Linear(in_dim, k)  # residual: 539 -> 64
        self.norm = nn.LayerNorm(k)

    def forward(self, clip, cat_features, log_ref_price):
        cat_embs = [emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_concat = torch.cat(cat_embs, dim=-1)
        # log_ref_price: [N, 1] or [N]
        if log_ref_price.dim() == 1:
            log_ref_price = log_ref_price.unsqueeze(-1)
        x = torch.cat([clip, cat_concat, log_ref_price], dim=-1)
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h) + self.skip(x)
        h = self.norm(h)
        return F.normalize(h, dim=-1)


class UserTowerV2(nn.Module):
    """ID(64) + Cat(6) + Cont(7) = 77D -> 64D L2-normed."""
    def __init__(self, n_users, k, cat_vocab_sizes, cat_dims, n_continuous, dropout=0.1):
        super().__init__()
        self.id_emb = nn.Embedding(n_users, k)
        nn.init.normal_(self.id_emb.weight, std=0.01)

        self.user_cat_embeddings = nn.ModuleList([
            nn.Embedding(vs, d) for vs, d in zip(cat_vocab_sizes, cat_dims)
        ])
        total_cat_dim = sum(cat_dims)
        in_dim = k + total_cat_dim + n_continuous  # 64 + 6 + 7 = 77

        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, k)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, cat_features, cont_features):
        id_e = self.id_emb(ids)
        cat_embs = [emb(cat_features[:, i]) for i, emb in enumerate(self.user_cat_embeddings)]
        cat_concat = torch.cat(cat_embs, dim=-1)
        x = torch.cat([id_e, cat_concat, cont_features], dim=-1)
        x = self.norm(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return F.normalize(x, dim=-1)


class TwoTowerInfoNCEv2(nn.Module):
    def __init__(self, n_users, clip_dim, k, item_cat_vocabs, item_cat_dims,
                 user_cat_vocabs, user_cat_dims, n_user_cont, dropout=0.1):
        super().__init__()
        self.user_tower = UserTowerV2(
            n_users, k, user_cat_vocabs, user_cat_dims, n_user_cont, dropout
        )
        self.item_tower = ItemTowerV2(
            clip_dim, k, item_cat_vocabs, item_cat_dims, dropout
        )

    def forward(self, user_ids, user_cat, user_cont, clip_feats, item_cat):
        return (
            self.user_tower(user_ids, user_cat, user_cont),
            self.item_tower(clip_feats, item_cat),
        )


model = TwoTowerInfoNCEv2(
    N_users, clip_features.shape[1], K,
    item_cat_vocab_sizes, ITEM_CAT_DIMS,
    user_cat_vocab_sizes, USER_CAT_DIMS,
    N_USER_CONT, DROPOUT,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
item_in = clip_features.shape[1] + sum(ITEM_CAT_DIMS) + 1
user_in = K + sum(USER_CAT_DIMS) + N_USER_CONT
log(f"Parameters: {n_params:,}")
log(f"Item: CLIP({clip_features.shape[1]}) + Cat({sum(ITEM_CAT_DIMS)}) + Price(1) = {item_in}d "
    f"-> Linear({item_in},256) -> GELU -> Drop -> Linear(256,{K}) + Skip({item_in},{K}) -> LN -> L2")
log(f"User: Emb({N_users},{K}) + Cat({sum(USER_CAT_DIMS)}) + Cont({N_USER_CONT}) = {user_in}d "
    f"-> LN -> Linear({user_in},{K}) -> GELU -> Drop -> L2")

# Move feature tensors to device
clip_tensor = torch.tensor(clip_features, dtype=torch.float32, device=device)
item_cat_tensor = torch.tensor(item_cat_features, dtype=torch.long, device=device)
user_cat_tensor = torch.tensor(user_cat_features, dtype=torch.long, device=device)
user_cont_tensor = torch.tensor(user_cont_features, dtype=torch.float32, device=device)
log_ref_price_tensor = torch.tensor(log_ref_prices, dtype=torch.float32, device=device)

# ════════════════════════════════════════════════════════════════════
# OPTIMIZER + CHECKPOINT RESUME
# ════════════════════════════════════════════════════════════════════
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

start_epoch = 0
best_hit10 = 0.0
best_epoch = 0
patience_ctr = 0

if CHECKPOINT_PATH.exists():
    log("\n-- Resuming from checkpoint --")
    ckpt = torch.load(str(CHECKPOINT_PATH), weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"]
    best_hit10 = ckpt["best_hit10"]
    best_epoch = ckpt["best_epoch"]
    patience_ctr = ckpt["patience_ctr"]
    rng.set_state(ckpt["rng_state"])
    log(f"Resumed at epoch {start_epoch}, best_hit10={best_hit10:.4f} (epoch {best_epoch}), "
        f"patience={patience_ctr}/{PATIENCE}")

# ════════════════════════════════════════════════════════════════════
# TRAINING — Full-Corpus InfoNCE with enriched features
# ════════════════════════════════════════════════════════════════════
n_train_batches = (len(train_pairs) + BATCH_SIZE - 1) // BATCH_SIZE

log(f"\n-- Training (epochs {start_epoch+1}-{EPOCHS}, InfoNCE, tau={TAU}, full catalog={N_items}) --")
log(f"{'Ep':>3} {'TrLoss':>8} {'VaLoss':>8} {'Hit@10':>8} {'NDCG@10':>8} {'LR':>10}")
log("-" * 55)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    perm = rng.permutation(len(train_pairs))
    train_shuf = train_pairs[perm]
    ep_loss = 0.0
    n_batch = 0

    # Pre-compute ALL item embeddings once per epoch — WITH gradient
    all_item_emb = model.item_tower(clip_tensor, item_cat_tensor, log_ref_price_tensor)  # [N_items, 64]

    pbar = tqdm(
        range(0, len(train_shuf), BATCH_SIZE),
        desc=f"Ep {epoch+1:2d}/{EPOCHS}",
        file=sys.stderr,
        ncols=80,
        leave=False,
    )
    for start in pbar:
        batch = train_shuf[start:start + BATCH_SIZE]
        uid = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
        pos_idx = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

        # User tower with enriched features
        user_emb = model.user_tower(
            uid, user_cat_tensor[uid], user_cont_tensor[uid]
        )
        logits = user_emb @ all_item_emb.T / TAU     # [B, N_items]
        loss = F.cross_entropy(logits, pos_idx)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Re-compute item embeddings after param update
        all_item_emb = model.item_tower(clip_tensor, item_cat_tensor, log_ref_price_tensor)

        ep_loss += loss.item()
        n_batch += 1
        pbar.set_postfix(loss=f"{ep_loss/n_batch:.4f}")

    pbar.close()
    train_loss = ep_loss / n_batch
    scheduler.step()

    # ── Validation ──
    model.eval()
    with torch.no_grad():
        all_item_emb_val = model.item_tower(clip_tensor, item_cat_tensor, log_ref_price_tensor)

        # Val InfoNCE loss
        vl_sum, vl_n = 0.0, 0
        for start in range(0, len(val_pairs), BATCH_SIZE):
            vb = val_pairs[start:start + BATCH_SIZE]
            uid_v = torch.tensor(vb[:, 0], dtype=torch.long, device=device)
            pos_v = torch.tensor(vb[:, 1], dtype=torch.long, device=device)
            ue_v = model.user_tower(uid_v, user_cat_tensor[uid_v], user_cont_tensor[uid_v])
            logits_v = ue_v @ all_item_emb_val.T / TAU
            vl = F.cross_entropy(logits_v, pos_v, reduction='sum')
            vl_sum += vl.item()
            vl_n += len(vb)
        val_loss = vl_sum / vl_n

        # Hit@10, NDCG@10 (full-corpus ranking)
        USER_CHUNK = 1024
        hits, ndcg_sum = 0, 0.0
        for start in range(0, len(val_pairs), USER_CHUNK):
            vb = val_pairs[start:start + USER_CHUNK]
            uid_v = torch.tensor(vb[:, 0], dtype=torch.long, device=device)
            true_j = vb[:, 1]
            ue_v = model.user_tower(uid_v, user_cat_tensor[uid_v], user_cont_tensor[uid_v])
            scores = ue_v @ all_item_emb_val.T
            top_k = scores.topk(10, dim=1).indices.cpu().numpy()
            for i, tj in enumerate(true_j):
                where = np.where(top_k[i] == tj)[0]
                if len(where) > 0:
                    hits += 1
                    ndcg_sum += 1.0 / np.log2(where[0] + 2)
        hit10 = hits / len(val_pairs)
        ndcg10 = ndcg_sum / len(val_pairs)

    # ── Early stopping on Hit@10 ──
    lr_now = scheduler.get_last_lr()[0]
    mark = ""
    if hit10 > best_hit10:
        best_hit10 = hit10
        best_epoch = epoch + 1
        patience_ctr = 0
        torch.save(model.state_dict(), str(ARTIFACTS_DIR / "model_state.pt"))
        mark = " *"
    else:
        patience_ctr += 1

    line = (f"{epoch+1:3d} {train_loss:8.4f} {val_loss:8.4f} "
            f"{hit10:8.4f} {ndcg10:8.4f} {lr_now:10.2e}{mark}")
    log(line)

    # ── Save checkpoint ──
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_hit10": best_hit10,
        "best_epoch": best_epoch,
        "patience_ctr": patience_ctr,
        "rng_state": rng.get_state(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "hit10": hit10,
        "ndcg10": ndcg10,
    }, str(CHECKPOINT_PATH))

    if patience_ctr >= PATIENCE:
        log(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
        break

log(f"\nBest: epoch {best_epoch}, Hit@10={best_hit10:.4f}")

# ════════════════════════════════════════════════════════════════════
# SAVE FINAL ARTIFACTS
# ════════════════════════════════════════════════════════════════════
log("\n-- Saving final artifacts --")

# Load best model
model.load_state_dict(
    torch.load(str(ARTIFACTS_DIR / "model_state.pt"), weights_only=True, map_location=device)
)
model.eval()

with torch.no_grad():
    # Chunk user embedding extraction (50K at a time)
    USER_EMB_CHUNK = 50000
    user_emb_parts = []
    for u_start in range(0, N_users, USER_EMB_CHUNK):
        u_end = min(u_start + USER_EMB_CHUNK, N_users)
        ids = torch.arange(u_start, u_end, device=device)
        user_emb_parts.append(
            model.user_tower(
                ids, user_cat_tensor[ids], user_cont_tensor[ids]
            ).cpu().numpy()
        )
    user_embs = np.concatenate(user_emb_parts, axis=0)
    item_embs = model.item_tower(clip_tensor, item_cat_tensor, log_ref_price_tensor).cpu().numpy()

np.save(str(ARTIFACTS_DIR / "user_embeddings.npy"), user_embs)
np.save(str(ARTIFACTS_DIR / "item_embeddings.npy"), item_embs)

# Save tower config for downstream scripts (lockdown extension)
tower_config = {
    "has_price_feature": True,
    "input_dim": 539,
    "clip_dim": int(clip_features.shape[1]),
    "cat_dim": int(sum(ITEM_CAT_DIMS)),
    "k": K,
    "dropout": DROPOUT,
    "item_cat_vocab_sizes": item_cat_vocab_sizes,
    "item_cat_dims": ITEM_CAT_DIMS,
}
with open(str(ARTIFACTS_DIR / "tower_config.json"), "w") as f:
    json.dump(tower_config, f, indent=2)

log(f"user_embeddings.npy  {user_embs.shape}")
log(f"item_embeddings.npy  {item_embs.shape}")
log(f"tower_config.json    has_price_feature=True, input_dim=539")
log(f"ref_prices.npy       {ref_prices.shape}")
log(f"model_state.pt       epoch {best_epoch}")
log(f"checkpoint.pt        full training state")

# Embedding stats
u_norms = np.linalg.norm(user_embs, axis=1)
i_norms = np.linalg.norm(item_embs, axis=1)
log(f"\nUser emb norms: mean={u_norms.mean():.4f}, std={u_norms.std():.4f}")
log(f"Item emb norms: mean={i_norms.mean():.4f}, std={i_norms.std():.4f}")

# RankMe for item embeddings
S = np.linalg.svd(item_embs, compute_uv=False)
p = S / S.sum()
p = p[p > 1e-10]
rankme = np.exp(-np.sum(p * np.log(p)))
log(f"RankMe (item embeddings): {rankme:.1f}")

train_time = time.time() - t0
log(f"\n{'=' * 80}")
log("ARM A TRAINING COMPLETE")
log(f"{'=' * 80}")
log(f"Time: {train_time:.1f}s ({train_time/60:.1f}m)")
log(f"Best: epoch {best_epoch}, Hit@10={best_hit10:.4f}")
log(f"End: {datetime.now()}")
log_file.close()
