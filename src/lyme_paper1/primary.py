"""Primary forecast comparison for the 174-unit cohort."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def run(inputs: Path, output: Path):
    data = pd.read_csv(inputs / "units.csv").sort_values(["outcome_year", "Plot"])
    if len(data) != 174 or not (data.outcome_year == data.source_year + 1).all():
        raise ValueError('Expected 174 plot-years paired with next-year observations.')
    plots = sorted(data.Plot.unique())
    def design(d, burden):
        cols = [np.log1p(d.mouse), np.log1p(d.chip), np.log1p(450*d.larvae), d.past_gdd]
        if burden:
            cols += [np.log1p(d.fed_m), np.log1p(d.fed_c)]
        cols += [(d.Plot == x).astype(float) for x in plots[1:]]
        return np.column_stack(cols)
    rows = []
    for year in range(2002, 2025):
        train = data[data.outcome_year < year]
        test = data[data.outcome_year == year]
        if train.outcome_year.nunique() < 10:
            raise ValueError("Insufficient past-only training history.")
        for label, burden in [("baseline", False), ("with_burden", True)]:
            x, z = design(train, burden), design(test, burden)
            mu, sd = x.mean(0), x.std(0)
            sd[sd < 1e-10] = 1
            x = np.column_stack([np.ones(len(x)), (x-mu)/sd])
            z = np.column_stack([np.ones(len(z)), (z-mu)/sd])
            penalty = np.eye(x.shape[1]); penalty[0,0] = 0
            coef = np.linalg.solve(x.T@x + penalty, x.T@np.log1p(train.outcome))
            pred = np.maximum(np.expm1(z@coef), 0)
            for r, v in zip(test.itertuples(), pred):
                rows.append(dict(year=year, Plot=r.Plot, model=label,
                                 observed=float(r.outcome), predicted=float(v),
                                 absolute_error=float(abs(v-r.outcome)),
                                 train_last_year=int(train.outcome_year.max())))
    pred = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "predictions.csv", index=False)
    metrics = []
    for label, group in pred.groupby("model", sort=True):
        metrics.append(dict(model=label, n_units=len(group), n_years=group.year.nunique(),
                            pooled_MAE=float(group.absolute_error.mean()),
                            year_equal_MAE=float(group.groupby("year").absolute_error.mean().mean())))
    table = pd.DataFrame(metrics); table.to_csv(output / "metrics.csv", index=False)
    m = table.set_index("model")
    summary = dict(pooled_relative_improvement=1-m.loc["with_burden","pooled_MAE"]/m.loc["baseline","pooled_MAE"],
                   year_equal_delta=float(m.loc["with_burden","year_equal_MAE"]-m.loc["baseline","year_equal_MAE"]),
                   strict_past_only_training=bool((pred.train_last_year < pred.year).all()),
                   scope='Predict next-year nymph density from host abundance, larval supply, weather and host burden.')
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return dict(metrics=metrics, **summary)
