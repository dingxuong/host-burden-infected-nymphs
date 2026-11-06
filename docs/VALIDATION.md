# Expected results

| Calculation | Result |
|---|---|
| Primary cohort | 174 complete plot-years; 136 test plot-years in 23 years |
| Primary pooled MAE | 19.905330352884345 without burden; 14.581433566465401 with burden |
| Dispersion bootstrap | 400 of 400 fits converged |
| Dispersion slope | 0.20342759662907095; 95% percentile interval [0.03576170317019307, 0.29494246508918653] |
| Annual burden-abundance slope | −0.2651685402885098 |
| Slope with year fixed effects | 0.022229672072569567 |
| Monotone-response problems | Guaranteed decrease in 34 of 108 cases |
| Concave-response problems | Guaranteed decrease in 18 of 36 cases |
| S3 infection comparison | 21 fitted models; 90 predictions |

The infection predictions agree with the supplied comparison values to within 2×10⁻¹⁵. Independent site-ranking checks cover 300,000 draws across 15 groups and 126 site records. Primal-dual checks pass for the response-bound and S20 calculations.

Calculations were checked with Python 3.9.23 and the library versions in `requirements.txt`. Installation, the exact example and command help were also checked with Python 3.11.16. The installation instructions recommend Python 3.10 or newer.

The exact example can be checked with `python -m unittest discover -s tests -v`. See [analysis commands](COVERAGE.md) to calculate the remaining results.
