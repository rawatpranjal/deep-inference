# RieszNet core port: known-truth coverage

Ported `riesz_inference`, M=40 replications per family.

| family | truth | mean est | bias | emp SE | mean SE | SE-ratio | coverage |
|---|---|---|---|---|---|---|---|
| linear | 0.5012 | 0.5052 | +0.0041 | 0.0591 | 0.0565 | 0.96 | 95% |
| logit | 0.1812 | 0.1807 | -0.0006 | 0.0253 | 0.0230 | 0.91 | 92% |
