'Propagate infection-count uncertainty through temporal and spatial risk summaries at fixed DON.'
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pcr_scale_uncertainty_2026-10-13'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'results/cary_transmission_paths_2026-10-10/modern_units.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(source)
d=pd.read_csv(source).sort_values(['outcome_year','Plot']).reset_index(drop=True)
assert len(d)==54 and d.outcome_year.nunique()==9 and d.Plot.nunique()==6
assert (d.n>0).all() and (d.k>=0).all() and (d.k<=d.n).all()
assert ((d.k%1)==0).all() and ((d.n%1)==0).all() and (d.DON_mean>0).all()
assert not d.duplicated(['outcome_year','Plot']).any()
assert np.max(abs(d.NIP-d.k/d.n))<1e-12
q=np.log(d.DON_mean.to_numpy())
years=sorted(d.outcome_year.unique())
iy=[np.flatnonzero(d.outcome_year.to_numpy()==y) for y in years]
left=[];right=[]
for plot,g in d.groupby('Plot'):
    g=g.sort_values('outcome_year')
    for a,b in zip(g.index[:-1],g.index[1:]):
        assert d.loc[b,'outcome_year']-d.loc[a,'outcome_year']==1
        left.append(a);right.append(b)
left=np.asarray(left);right=np.asarray(right)
assert len(left)==48
def fraction(a,b):
    ac=a-np.mean(a,axis=-1,keepdims=True)
    bc=b-np.mean(b,axis=-1,keepdims=True)
    return (np.mean(bc*bc,axis=-1)+np.mean(ac*bc,axis=-1))/np.var(a+b,axis=-1)
def both(p):
    lp=np.log(p)
    st=fraction(q[right]-q[left],lp[...,right]-lp[...,left])
    qp=q.copy();pp=lp.copy()
    for ids in iy:
        qp[ids]-=qp[ids].mean()
        pp[...,ids]-=pp[...,ids].mean(axis=-1,keepdims=True)
    return st,fraction(qp,pp)
observed=both(d.NIP.to_numpy())
assert np.max(abs(np.asarray(observed)-[0.1297132260166913,0.3667162537960093]))<1e-12
rng=np.random.default_rng(2026101307)
p=rng.beta(d.k.to_numpy()+.5,d.n.to_numpy()-d.k.to_numpy()+.5,size=(20000,len(d)))
time,space=both(p);delta=space-time
draws=pd.DataFrame({'draw':np.arange(1,20001),'temporal_fraction':time,'spatial_fraction':space,'spatial_minus_temporal':delta})
assert np.isfinite(draws.to_numpy()).all()
draws.to_csv(OUT/'paired_scale_draws.csv',index=False)
d[['outcome_year','Plot','k','n','DON_mean','NIP']].to_csv(OUT/'all_54_input_units.csv',index=False)
def summary(x):
    return {'median':float(np.median(x)),'conditional95':np.quantile(x,[.025,.975]).tolist()}
result={'n_units':54,'n_transitions':48,'n_tested':int(d.n.sum()),'n_positive':int(d.k.sum()),
    'observed_temporal_fraction':float(observed[0]),'observed_spatial_fraction':float(observed[1]),
    'temporal_fraction':summary(time),'spatial_fraction':summary(space),
    'paired_difference':summary(delta),'conditional_probability_spatial_greater':float(np.mean(delta>0)),
    'passes_fixed_positive_interval_rule':bool(np.quantile(delta,.025)>0),
    'interpretation':'Uncertainty from PCR counts, conditional on observed DON; forest-sampling uncertainty is excluded.',
    'source':str(source),'source_sha256':before,
    'seed':2026101307,'draws':20000}
assert sha(source)==before
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
