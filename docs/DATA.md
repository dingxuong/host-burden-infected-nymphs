# Data sources and variables

## Sources

| Dataset | Provider DOI | Role |
|---|---|---|
| Cary long-term host and tick records | https://doi.org/10.25390/caryinstitute.31872853 | Annual ecological inputs and next-year nymph outcomes |
| Cary modern host/tick/pathogen records | https://doi.org/10.25390/caryinstitute.27871269 | Seasonal density, infection counts and within-year priorities |
| Capture–recapture abundance | https://doi.org/10.25390/caryinstitute.25742778 | Alternative abundance representations |
| Individual attached larval burdens | https://doi.org/10.25390/caryinstitute.25747119 | Burden aggregation and observed distribution shapes |
| Scottish forests, Gandy et al. | https://doi.org/10.5061/dryad.rv15dv49r | Independent risk-component and site-selection example |
| Daymet V4 R1 | https://doi.org/10.3334/ORNLDAAC/2129 | Seasonal weather predictors |

Observational data are obtained from the providers above and remain subject to their terms. This repository contains analysis code and aggregate reference metrics.

## Data files

The accompanying `paper1-inputs` directory contains:

```text
paper1-inputs/
  manifest.json
  forecast/             Annual ecological data, forecasts and infection observations
  selection/inputs/     Cary and Scottish site observations
  theory/               Burden distributions and response-bound solutions
  interval_workspace/   Weighted burden distributions for response bounds
```

`manifest.json` lists the files and SHA-256 hashes used by the main commands. `supplementary_manifest.json` lists those used by `reproduce`. Run `check-data` to check the main-command files.

`forecast/units.csv` contains 174 plot-years, pairing host measurements in year t−1 with nymph density in year t. Abundance multiplied by attached larval burden is an attachment index. Column definitions are in [the variable dictionary](input_column_dictionary.csv).

The `primary` and `diagnostics` commands predict next-year nymph density. Scottish inputs support risk-component and site-selection comparisons.

## Paired and experimental observations

| Dataset | Provider DOI | Fields used |
|---|---|---|
| Norwegian small mammals, Lindsø | https://doi.org/10.5281/zenodo.10695801 | `ID_long`, `NN_larva`, `Bburg2`, `year_cat`, `sp_eng`, `season` in the source semicolon-delimited CSV |
| Jacquet source-host experiment | https://doi.org/10.1186/s13071-016-1929-z | Supplementary workbook Sheet2: host ID, sampling occasion, treatment and nymph assay ID |

The Norwegian analysis retains 1,391 unique hosts, all zero larval counts and the original tissue marker; a separate sensitivity applies the authors' count replacement. The Jacquet inventory concerns 1,174 selected nymph assay records nested in 23 infected mice. 

## Supplementary data

The supplementary data are under `supplementary/lyme/`:

- `data/raw/`: original Cary tables, capture histories, modern infection tables, Norwegian records and raw Daymet station-point data.
- `literature/`: the Jacquet workbook and earlier Cary table.
- `results/`: annual ecological tables, model predictions and S3 abundance distributions.
- `reviews/`: weighted burden distributions and hypothetical measurement intervals.
- `manuscript/`: numerical tables used by the analysis scripts.

Use `reproduce build-annual-inputs` to aggregate the annual ecological tables, `reproduce shapes` for burden distributions and `reproduce environment` for seasonal weather. `reproduce paired-observations` reads the Norwegian CSV and Jacquet workbook. The main forecasting and selection commands read `forecast/` and `selection/`. Required filenames and columns are specified in each script; see [analysis commands](COVERAGE.md).

Use the source identifiers above when citing the data, and follow each provider's access and reuse terms.
