"""
Auto-research v4: stability check + honest MC inflation.

Observation: Round 1 C12 got 0.900 coverage; Round 2 C17 (same arch, same
ensemble, more patience) got 0.733.  Hypothesis: this is MC noise, not a
methodological difference.

Tests:
  S1: C12 config, 5 seed offsets × M=15 each.  Measure coverage SD across
      seed sets to estimate true MC noise at M=15.
  S2: C12 config, M=60 (4x more reps).  Should stabilize coverage to true
      population value.
  S3: C12 config, M=100.  Near-ground-truth coverage estimate.

If S3 says the true coverage is ~0.85 (not 0.90), round-1 C12 was lucky.
If S3 says ~0.90, round-2 was unlucky.
Either way, we want to know the TRUE coverage with tight CIs.
"""

import numpy as np
import time
import json
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_autoresearch_v2 import (
    N_CELLS, N_WORKERS, FINDINGS_PATH,
    MU_CELL_STAR,
)
from flm_autoresearch_v3 import run_one_generic, run_config_generic


BASE_CFG = dict(arch="tanh", hidden=(128, 64), ensemble_r=5)


def run_with_seed_offset(offset, M=15, cfg=BASE_CFG):
    t0 = time.time()
    print(f"  [seed_offset={offset}  M={M}]")
    payloads = [("C12_stab", offset + m * 31, 40_000, cfg) for m in range(M)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one_generic, p) for p in payloads]
        results = [f.result() for f in as_completed(futs)]
    wall = time.time() - t0

    per_cell_cov = []
    for c in range(N_CELLS):
        mu = np.array([r[f"fml_c{c}_mu"] for r in results])
        se = np.array([r[f"fml_c{c}_se"] for r in results])
        v = ~np.isnan(mu) & ~np.isnan(se)
        if v.sum() < 3:
            continue
        cov = float(np.mean(np.abs(mu[v] - MU_CELL_STAR[c]) <= 1.96 * se[v]))
        per_cell_cov.append(cov)
    return dict(offset=offset, M=M, wall=wall,
                avg_cov=float(np.mean(per_cell_cov)),
                cells_ge_90=int(sum(c >= 0.90 for c in per_cell_cov)),
                cells_lt_70=int(sum(c < 0.70 for c in per_cell_cov)))


def main():
    t_all = time.time()

    with open(FINDINGS_PATH, "a") as f:
        f.write(f"\n\n## Round 3: Stability check  "
                f"(t={datetime.datetime.now().isoformat(timespec='seconds')})\n\n"
                "Rerun C12 config (Tanh ensemble-5) with 5 different seed "
                "offsets to measure MC noise at M=15.  Then run M=60 to get "
                "a tight estimate of true coverage.\n")

    # S1: same config, 5 different seed starts, M=15 each
    print("\n=== S1: seed stability at M=15 ===")
    s1 = []
    for offset in [10_000, 20_000, 30_000, 40_000, 50_000]:
        r = run_with_seed_offset(offset, M=15)
        print(f"    offset={offset}  avg_cov={r['avg_cov']:.4f}  "
              f"ge90={r['cells_ge_90']}  lt70={r['cells_lt_70']}  "
              f"wall={r['wall']:.0f}s")
        s1.append(r)
    covs = [r["avg_cov"] for r in s1]
    print(f"\n  S1 across 5 seed sets:")
    print(f"    mean coverage across seed sets: {np.mean(covs):.4f}")
    print(f"    std  coverage across seed sets: {np.std(covs, ddof=1):.4f}")
    print(f"    range: [{min(covs):.3f}, {max(covs):.3f}]")

    # S2: one big run M=60
    print(f"\n=== S2: high-M stability test at M=60 ===")
    s2 = run_with_seed_offset(60_000, M=60)
    print(f"    M=60  avg_cov={s2['avg_cov']:.4f}  "
          f"ge90={s2['cells_ge_90']}  lt70={s2['cells_lt_70']}  "
          f"wall={s2['wall']:.0f}s")

    # S3: another big run M=60 with different seed
    print(f"\n=== S3: second high-M test M=60 ===")
    s3 = run_with_seed_offset(70_000, M=60)
    print(f"    M=60  avg_cov={s3['avg_cov']:.4f}  "
          f"ge90={s3['cells_ge_90']}  lt70={s3['cells_lt_70']}  "
          f"wall={s3['wall']:.0f}s")

    # append to FINDINGS.md
    with open(FINDINGS_PATH, "a") as f:
        f.write("\n### S1: MC noise at M=15 across 5 seed offsets\n\n")
        f.write("| seed_offset | avg_cov | cells ≥ 0.90 | cells < 0.70 | wall |\n")
        f.write("|---|---|---|---|---|\n")
        for r in s1:
            f.write(f"| {r['offset']} | {r['avg_cov']:.4f} | "
                    f"{r['cells_ge_90']}/40 | {r['cells_lt_70']}/40 | "
                    f"{r['wall']:.0f}s |\n")
        f.write(f"\n**MC noise at M=15**: "
                f"mean={np.mean(covs):.4f}, "
                f"SD={np.std(covs, ddof=1):.4f}, "
                f"range=[{min(covs):.3f}, {max(covs):.3f}]\n\n")

        f.write("### S2 + S3: high-M=60 stability\n\n")
        for s, label in [(s2, "S2"), (s3, "S3")]:
            f.write(f"- {label}: M=60, avg_cov={s['avg_cov']:.4f}, "
                    f"cells≥0.90={s['cells_ge_90']}/40, "
                    f"cells<0.70={s['cells_lt_70']}/40, wall={s['wall']:.0f}s\n")

        combined_cov = (s2["avg_cov"] + s3["avg_cov"]) / 2
        f.write(f"\n**True coverage estimate (averaged across S2 + S3, "
                f"M=120 effective): {combined_cov:.4f}**\n")

    print(f"\n[v4 total] {time.time()-t_all:.0f}s")
    print(f"[v4] findings → {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
