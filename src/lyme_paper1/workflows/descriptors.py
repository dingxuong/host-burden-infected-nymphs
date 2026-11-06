'Summarize burden-distribution shapes and prediction errors by year.'
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'lyme/manuscript/paper1/clean_review_copy_2026-10-22'
SHAPE = ROOT / 'lyme/reviews/2026-10-14/empirical_shape_public_v1'
PRED = ROOT / 'lyme/manuscript/paper1/ecology_letters_evidence_figures_refs_2026-10-14/source_data/frozen_empirical/fig2_all_136_matched_predictions.csv'
SCOT = ROOT / 'lyme/manuscript/paper1/full_revision_2026-10-17/review_materials/fixed_selection/inputs/scotland.csv'
inputs = [SHAPE/'animal_observation_means.csv', SHAPE/'shape_group_summary.csv', PRED, SCOT]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
(OUT/'INPUT_SHA256.json').write_text(json.dumps(hashes, indent=2))
animals = pd.read_csv(inputs[0]); original = pd.read_csv(inputs[1])
animals = animals[animals.version.eq('animal_augsep')]
rows = []
for (sp, year, grid), group in animals.groupby(['species', 'year', 'grid']):
    x = group.mean_observed_burden.to_numpy(float)
    if len(x) < 10 or x.mean() <= 0:
        continue
    z = x/x.mean(); sd = z.std(ddof=0)
    row = dict(species=sp, year=int(year), grid=grid, n=len(x), mean=x.mean(),
               cv=sd, skew=float(np.mean((z-1)**3)/sd**3) if sd else float('nan'),
               zero_fraction=float(np.mean(x == 0)),
               top_decile_share=float(np.sort(x)[-math.ceil(.1*len(x)):].sum()/x.sum()),
               max_normalized_z=float(z.max()))
    old = original[(original.version == 'animal_augsep') & (original.species == sp)
                   & (original.year == year) & (original.grid == grid)]
    assert len(old) == 1
    for new, oldkey in [('cv','normalized_cv'),('zero_fraction','zero_fraction'),
                        ('top_decile_share','top_decile_share'),('max_normalized_z','max_normalized_z')]:
        assert abs(row[new]-float(old.iloc[0][oldkey])) < 1e-11, (new, sp, year, grid)
    rows.append(row)
units = pd.DataFrame(rows)
assert units.groupby('species').size().to_dict() == {'PL':181, 'TS':141}
units.to_csv(OUT/'observed_shape_unit_descriptors.csv', index=False)
summaries = []
for sp, group in units.groupby('species'):
    for metric in ['cv','skew','zero_fraction','top_decile_share','max_normalized_z']:
        values = group[metric].to_numpy()
        q = np.quantile(values, [0,.05,.5,.95,1], method='linear')
        summaries.append(dict(species=sp, metric=metric, n=len(values), minimum=q[0],
                              p05=q[1], median=q[2], p95=q[3], maximum=q[4]))
pd.DataFrame(summaries).to_csv(OUT/'observed_shape_descriptor_summary.csv', index=False)

d = pd.read_csv(PRED)
wide = d.pivot(index=['outcome_year','Plot'],columns='model',values='predicted')
obs = d.groupby(['outcome_year','Plot'])['observed'].agg(['min','max','first'])
assert np.max(obs['max']-obs['min']) == 0
wide['observed'] = obs['first']
for name in ['host_only','fed_proxy']:
    wide[name+'_ae'] = abs(wide[name]-wide.observed)
    saved = d[d.model.eq(name)].set_index(['outcome_year','Plot']).absolute_error
    assert np.max(abs(wide[name+'_ae']-saved)) < 1e-12
wide['delta'] = wide.fed_proxy_ae-wide.host_only_ae
annual = wide.groupby(level=0).mean(numeric_only=True)
annual.to_csv(OUT/'annual_fixed_forecast_audit.csv')
forecast = {}
for label, frame in [('pooled_136',wide), ('year_equal_23',annual),
                     ('without_2024_pooled',wide.loc[wide.index.get_level_values(0)!=2024]),
                     ('without_2024_year_equal',annual.loc[annual.index!=2024])]:
    a,b = frame[['host_only_ae','fed_proxy_ae']].mean()
    forecast[label] = dict(n=len(frame), baseline=float(a), burden=float(b),
                           delta=float(b-a), relative_reduction=float(1-b/a))
forecast['observed_peak_DON'] = dict(zip(['minimum','q25','median','q75','maximum'],
                                       map(float,np.quantile(wide.observed,[0,.25,.5,.75,1]))))
sc = pd.read_csv(SCOT)
sets = {int(y):set(g.Site_code) for y,g in sc.groupby('Year')}
scotland = dict(forests_per_year={str(y):len(s) for y,s in sets.items()},
               common_forest_ids=sorted(sets[2018] & sets[2019]),
               visits=len(sc), assays=int(sc.n.sum()))
report = dict(shape_summary=summaries, forecast=forecast, scotland=scotland,
              synthetic_AB=dict(mean_normalized=1, cv=.5, skew=0,
                                A_support=[.5,1.5],A_probabilities=[.5,.5],
                                B_support=[0,1,2],B_probabilities=[.125,.75,.125]),
              scope='Burden-distribution descriptors and weighted errors from the primary forecasts.')
(OUT/'DESCRIPTIVE_AUDIT.json').write_text(json.dumps(report,indent=2))

