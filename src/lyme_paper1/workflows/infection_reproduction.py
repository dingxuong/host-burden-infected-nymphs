'Fit the three S3 infection models and calculate infected-nymph density.\n\nUse the same 190-component abundance prediction distribution for all three\ninfection-dispersion settings.\n'
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.special import logsumexp
from infection_functions import GRID, configure, name, fit, observation, cdf_and_atoms, risk_median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/infection_reproduction'
OUT.mkdir(parents=True, exist_ok=True)
DATA = pd.read_csv(ROOT / 'results/cary_transmission_paths_2026-10-10/modern_units.csv')
configure(DATA, OUT)
records=[];predictions=[];numerics=[]
for year in range(2016,2023):
 te=DATA[DATA.outcome_year.eq(year)].sort_values('Plot')
 for phi in [120.,160.,np.inf]:
  c=('chip_pressure','iid_tight',phi,2.5);_,diag=fit(year,c,6);records.append(diag)
  if year<2018:continue
  a=dict(np.load(OUT/f'fit_{year}_{name(c)}_z6.npz'));a.update(kind='state',mass=1.)
  d=np.load(ROOT/f'results/contextual_amplitude_2026-10-12/density_{year}_constant0.5.npz')
  ll,_=observation(te.k.to_numpy()[None,:],te.n.to_numpy()[None,:],GRID[:,None]+a['offset'][0][None,:],phi)
  nll=-logsumexp(ll+np.log(np.maximum(a['state_weight'][0],1e-300))[:,None],axis=0)
  for j,row in enumerate(te.to_dict('records')):
   ccdf,atoms,aw,p,mass=cdf_and_atoms([a],j)
   mu,sd,w=d['loc'][:,j],np.sqrt(d['cov'][:,j,j]),d['weights']
   risk,tail=risk_median(ccdf,atoms,aw,mass,mu,sd,w,4097)
   if np.isfinite(phi):fine,tail=risk_median(ccdf,atoms,aw,mass,mu,sd,w,8193)
   else:fine=risk
   predictions.append(dict(year=year,Plot=row['Plot'],model=name(c),predicted_NIP=p,predicted_DIN=fine,count_NLL=nll[j],observed_NIP=row['k']/row['n'],observed_DIN=row['DIN_mean']))
   numerics.append(dict(year=year,Plot=row['Plot'],model=name(c),resolution_difference=abs(fine-risk),tail_mass=tail))
 print(year,'completed',flush=True)
p=pd.DataFrame(predictions)
for t in ['NIP','DIN']:p[t+'_AE']=abs(p['predicted_'+t]-p['observed_'+t])
p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(numerics).to_csv(OUT/'numerical_audit.csv',index=False)
m=p.groupby('model').agg(NIP_MAE=('NIP_AE','mean'),DIN_MAE=('DIN_AE','mean'),count_NLL=('count_NLL','mean'),n=('Plot','size'));m.to_csv(OUT/'metrics.csv')
ref=pd.read_csv(ROOT/'results/chip_dispersion_frontier_2026-10-12/predictions.csv')
a=p.merge(ref,on=['year','Plot','model'],suffixes=('','_reference'),validate='one_to_one');assert len(a)==90
checks={t:float(abs(a[t]-a[t+'_reference']).max()) for t in ['predicted_NIP','predicted_DIN','count_NLL']}
assert max(checks.values())<2e-5,checks
(OUT/'verification.json').write_text(json.dumps(dict(predictions=90,refits=len(records),maximum_differences=checks,scope='Fit infection models to modern counts and combine them with the supplied density prediction distributions.'),indent=2))
(OUT/'fit_diagnostics.json').write_text(json.dumps(records,indent=2));print(m.to_string(),flush=True)
