# Analysis definitions

A plot-year pairs host measurements in year t−1 with nymph density in year t. The primary cohort contains 174 complete plot-years; 136 are used for testing across 23 years. Each test year is predicted using earlier years. Pooled MAE gives each test plot-year equal weight; year-equal MAE first averages errors within a year and then averages across years.

Host abundance multiplied by mean attached larval burden measures attachment. The individual-burden analysis retains repeated captures; the observed-distribution analysis also calculates per-animal means. Their data definitions are given in the corresponding scripts.

The infection analysis uses the same predicted nymph-density distribution for all infection models. It fits infection counts under Beta-binomial or binomial observation models and integrates density and infection proportion to calculate infected-nymph density. The supplied density distributions and literature-based comparison predictions are fitted model outputs. The S3 command refits the infection models using these supplied predictions.

The A/B example uses exact fractions. The response-bound analyses calculate minimum and maximum output within specified response-function classes. These ranges describe uncertainty about the response function; ecological confidence intervals and hypothetical measurement intervals have separate meanings.

The site-selection analysis holds DON and the number of selected sites fixed. PCR uncertainty follows the Jeffreys infection-proportion model. Norwegian tissue positivity, larval attachment and Jacquet post-moult infection assays are distinct observation endpoints.

See [analysis commands](COVERAGE.md) and [data definitions](DATA.md). The input-file lists include SHA-256 hashes so that the data versions can be checked.
