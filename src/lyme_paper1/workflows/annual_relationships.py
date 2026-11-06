"""Annual burden–abundance slope with source-year bootstrap and year controls."""
from pathlib import Path
import ast,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/cary_compensation_forecast_2026-10-10'
u=pd.read_csv(P/'long_units.csv');plots=sorted(u.Plot.unique());rng=np.random.default_rng(20261011)
def slope(d,year_effects=False):
 a=[np.ones(len(d)),np.log(d.mouse),np.log1p(450*d.larvae),*[(d.Plot==p).astype(float) for p in plots[1:]]]
 if year_effects:a += [(d.source_year==y).astype(float) for y in sorted(d.source_year.unique())[1:]]
 return float(np.linalg.lstsq(np.column_stack(a),np.log(d.burden_m),rcond=None)[0][1])
years=sorted(u.source_year.unique());values=[]
for _ in range(3000):
 sy=rng.choice(years,len(years),replace=True);d=pd.concat([u[u.source_year==y] for y in sy],ignore_index=True);values.append(slope(d))
r={'annual_log_slope':slope(u),'bootstrap_interval95':np.quantile(values,[.025,.975]).tolist(),'year_fixed_effect_slope':slope(u,True),'resamples':3000,'scope':'Capture-mean burden, grid and larval-supply adjustment; source-year resampling.'}
(P/'annual_relationships.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
