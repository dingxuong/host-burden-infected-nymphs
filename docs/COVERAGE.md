# Analysis commands

Run `python -m lyme_paper1` followed by a command below. The `reproduce` commands take `--inputs ../paper1-inputs --output outputs/<name>`; other analysis commands take `--data ../paper1-inputs --output outputs/<name>`.

| Analysis | Command | Calculation |
|---|---|---|
| Primary forecast | `primary` | Fit the two ridge models and predict 136 future plot-years |
| Annual data from source tables | `prepare-primary` | Join source-year host data to next-year nymph observations; see README for arguments |
| A/B example | `demo` | Calculate matching moments and opposite changes in infection output using exact fractions |
| S12 forecasting models | `diagnostics` | Fit 11 regression, dynamic and spline models |
| S15 site selection | `selection` | Calculate coverage, expected gains and stability under repeated PCR counts |
| S15 numerical check | `verify-selection` | Recalculate site ranks and gains; add `--results outputs/selection` |
| S11 response bounds | `interval-bounds` | Calculate monotone and concave response bounds with 50-decimal interval arithmetic |
| S11 numerical check | `certificates` | Check the coefficients and primal-dual solutions of 36 concave-response problems |
| Annual ecological data | `reproduce build-annual-inputs` | Aggregate source tables and calculate annual model comparisons |
| S1 count models | `reproduce counts` | Fit negative-binomial and occurrence/intensity models to capture records |
| S1 dispersion | `reproduce dispersion-bootstrap` | Fit 400 bootstrap samples, resampling complete source years |
| S1 burden-abundance relationship | `reproduce annual-relationships` | Fit annual slopes, 3,000 year-cluster bootstrap samples and year fixed effects |
| S2 path and forecasting models | `reproduce count-path-models` | Fit annual path, mixed, heteroscedastic and neural-network models |
| S2 literature comparison | `reproduce matched-forecasts` | Compare the Ostfeld et al. model structure with ridge models on the same training and test rows |
| S4 historical alignment | `reproduce alignment` | Calculate 500 residual-year permutation controls and spatial-block/year holdouts |
| S9 forecasting sensitivity | `reproduce forecast-controls` | Compare burden summaries, abundance estimates, penalties and influential years |
| S9 weather and neural networks | `reproduce weather-ann` | Compare weather forecasts and five-year neural-network validation |
| S6 infection-count uncertainty | `reproduce pcr-scales` | Propagate PCR uncertainty through temporal and spatial summaries at fixed DON |
| S10 observed burden distributions | `reproduce shapes` | Clean capture records, calculate animal means and weighted distributions, and plot conditional output ranges |
| S13 weather comparison | `reproduce environment` | Aggregate daily Daymet data and compare forecasts with and without weather predictors |
| S18 distribution summaries | `reproduce descriptors` | Calculate distribution descriptors and reweight primary prediction errors |
| S19 paired observations | `reproduce paired-observations` | Summarize Norwegian host records and Jacquet assays; export Cary ECDF and Norwegian panel tables |
| S3 infection prediction | `reproduce infection-forecasts` | Fit 21 infection models and calculate 90 predictions using the supplied density distributions |
| S20 measurement conditions | `reproduce measurement` | Calculate 36 baseline cases, 72 thresholds and 432 hypothetical measurement scenarios |
| S20 uncertain anchors | `reproduce anchor-sensitivity` | Calculate 102 combinations of measurement strategies and anchor intervals |

For `reproduce`, tables and figures are under `outputs/<name>/workspace/lyme/`. The other commands write results directly under the specified output directory. `check-data` checks the input-file hashes.

The `shapes` calculation produces the animal-mean table used by `descriptors`. The `measurement` calculation produces the hypothetical intervals used by `anchor-sensitivity`. Copies of these intermediate tables are included with the data so each analysis can run separately.

The S3 command takes the 190-component density distributions as inputs and refits the infection models. The abundance distributions and literature-based predictions supplied with the data are fitted model outputs.
