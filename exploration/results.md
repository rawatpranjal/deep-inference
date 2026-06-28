# Spike: FLM vs RieszNet (known-truth ATE)

### Linear DGP  (truth = 1.0000, M = 3)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 1.0652 | +0.0652 | 0.1245 | 0.0749 | 0.60 | 67% |
| FLM | 1.0678 | +0.0678 | 0.1265 | 0.0708 | 0.56 | 67% |
| RieszNet | 1.0299 | +0.0299 | 0.0979 | 0.0772 | 0.79 | 100% |
| Naive | 0.9963 | -0.0037 | 0.0894 | 0.0118 | 0.13 | 0% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
