# Larval allocation and biological responses shape infected-nymph production

Analysis code for the paper by Xudong Ding and Xinzhu Meng. It calculates next-year nymph forecasts, infection-informed site selection, host-burden distributions and bounds on infected-nymph production.

## Installation

Use Python 3.10 or newer. From this directory, run:

```bash
python -m pip install '.[full]'
```

The library versions used for the calculations are listed in `requirements.txt`.

The exact A/B example requires no observational data:

```bash
python -m lyme_paper1 demo
```

## Data

Unzip the accompanying data as `paper1-inputs` beside this directory. [Data sources and variables](docs/DATA.md) describes the files and their providers.

To prepare the primary forecast data directly from the two Cary source tables:

```bash
python -m lyme_paper1 prepare-primary --long-table /path/to/Full_dataset_20260326.csv --burdens /path/to/Larvae_onHosts_91-23.csv --data data-primary
python -m lyme_paper1 primary --data data-primary --output outputs/primary
```

These commands produce 174 complete plot-years and evaluate 136 future plot-years in 23 years. Pooled MAE decreases from 19.905330 to 14.581434 when host burdens are included.

## Analyses

With the accompanying data:

```bash
python -m lyme_paper1 primary --data ../paper1-inputs --output outputs/primary
python -m lyme_paper1 diagnostics --data ../paper1-inputs --output outputs/diagnostics
python -m lyme_paper1 selection --data ../paper1-inputs --output outputs/selection
python -m lyme_paper1 interval-bounds --data ../paper1-inputs --output outputs/bounds
python -m lyme_paper1 reproduce shapes --inputs ../paper1-inputs --output outputs/shapes
python -m lyme_paper1 reproduce paired-observations --inputs ../paper1-inputs --output outputs/paired
python -m lyme_paper1 reproduce infection-forecasts --inputs ../paper1-inputs --output outputs/infection
python -m lyme_paper1 reproduce measurement --inputs ../paper1-inputs --output outputs/measurement
python -m lyme_paper1 reproduce anchor-sensitivity --inputs ../paper1-inputs --output outputs/anchors
```

[Commands for all analyses](docs/COVERAGE.md) lists the remaining calculations. Choose a separate output directory for each analysis. For `reproduce`, calculated tables and figures are under `outputs/<analysis>/workspace/lyme/`.

The infection calculation fits models to the observed infection counts and combines them with the supplied 190-component nymph-density distributions. Literature-based predictions are included in the data for comparison.

See [analysis definitions](docs/REPRODUCIBILITY.md), [expected results](docs/VALIDATION.md) and the [中文说明](README.zh-CN.md).
