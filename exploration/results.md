# Spike: FLM vs RieszNet (known-truth ATE)

### Linear DGP (lambda=rf, alpha=1000, three_way=True, n=2000)  (truth = 1.0000, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9993 | -0.0007 | 0.0548 | 0.0535 | 0.98 | 98% |
| FLM | 1.0186 | +0.0186 | 0.0804 | 0.0918 | 1.14 | 98% |
| RieszNet | 0.9898 | -0.0102 | 0.0541 | 0.0544 | 1.01 | 94% |
| Naive | 0.9850 | -0.0150 | 0.0605 | 0.0109 | 0.18 | 26% |

**FLM parameter recovery** (theta_hat(x) vs truth, mean over reps)

| param | R2 | RMSE | bias |
|---|---|---|---|
| alpha(x) | 0.9215 | 0.2783 | +0.0272 |
| beta(x)  | 0.4258 | 0.3765 | -0.0148 |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
