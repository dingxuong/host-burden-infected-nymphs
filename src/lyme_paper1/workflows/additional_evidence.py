"""Fixed spatial/temporal transfer and burden-information controls."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=HERE/'additional_experiments';OUT.mkdir(exist_ok=True)
INPUT=ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv'
REF=ROOT/'results/cary_compensation_forecast_2026-10-10/forward_predictions.csv'
D=pd.read_csv(INPUT).sort_values(['outcome_year','Plot']).reset_index(drop=True)
reference=pd.read_csv(REF);reference=reference[reference.cohort.eq('long_peak_DON')&reference.model.isin(['host_only','fed_proxy'])].copy()
PLOTS=sorted(D.Plot.unique());YEARS=[int(y) for y in sorted(reference.outcome_year.unique())]
BLOCK={'GC':'Green','GE':'Green','HC':'Henry','HE':'Henry','TC':'Tea','TE':'Tea'}
D['block']=D.Plot.map(BLOCK)

def design(d,burden=False,full=False):
 cols=[np.log1p(d.mouse.to_numpy()),np.log1p(d.chip.to_numpy()),np.log1p(450*d.larvae.to_numpy()),d.past_gdd.to_numpy()]
 if burden:cols.extend([np.log1p((d.mouse*d.burden_m).to_numpy()),np.log1p((d.chip*d.burden_c).to_numpy())])
 cols.extend([(d.Plot==p).to_numpy().astype(float) for p in (PLOTS if full else PLOTS[1:])])
 return np.column_stack(cols)

def solve(X,Z,y):
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
 X=np.column_stack([np.ones(len(X)),(X-mu)/sd]);Z=np.column_stack([np.ones(len(Z)),(Z-mu)/sd])
 pen=np.eye(X.shape[1]);pen[0,0]=0
 return Z@np.linalg.solve(X.T@X+pen,X.T@y)

def predict(tr,te,burden=False,full=False):
 logp=solve(design(tr,burden,full),design(te,burden,full),np.log1p(tr.outcome.to_numpy()))
 return np.maximum(np.expm1(logp),0)

def block_interval(delta):
 v=np.asarray(delta);rng=np.random.default_rng(2026101302)
 st=rng.integers(0,len(v),(10000,int(np.ceil(len(v)/3))))
 ix=((st[:,:,None]+np.arange(3))%len(v)).reshape(10000,-1)[:,:len(v)]
 return np.quantile(v[ix].mean(1),[.025,.975]).tolist()

manifest={'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [INPUT,REF]],
 'years':YEARS,'fixed_alpha':1,'placebo_repetitions':500,'random_seed':2026101301,'new_external_data':False}
(OUT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

# Check the baseline predictions before calculating the controls.
repro=[]
for y in YEARS:
 tr=D[D.outcome_year<y];te=D[D.outcome_year.eq(y)]
 for key,flag in [('host_only',False),('fed_proxy',True)]:
  vv=predict(tr,te,flag)
  for r,v in zip(te.to_dict('records'),vv):repro.append({'outcome_year':y,'Plot':r['Plot'],'model':key,'predicted_reproduced':v})
rr=pd.DataFrame(repro).merge(reference[['outcome_year','Plot','model','predicted']],on=['outcome_year','Plot','model'],validate='one_to_one')
maxdiff=float(abs(rr.predicted-rr.predicted_reproduced).max())
rr.to_csv(OUT/'original_prediction_reproduction.csv',index=False)
if maxdiff>1e-8:raise RuntimeError(('Original predictions not reproduced',maxdiff))

spatial=[]
for y in YEARS:
 for block in ['Green','Henry','Tea']:
  tr=D[(D.outcome_year<y)&~D.block.eq(block)];te=D[D.outcome_year.eq(y)&D.block.eq(block)]
  if len(te)==0:continue
  if tr.outcome_year.nunique()<10:raise RuntimeError(('Insufficient training years',y,block))
  for key,flag in [('baseline',False),('burden',True)]:
   for r,v in zip(te.to_dict('records'),predict(tr,te,flag,True)):
    spatial.append({'year':y,'Plot':r['Plot'],'held_block':block,'model':key,'observed':r['outcome'],'predicted':v,
      'AE':abs(v-r['outcome']),'training_rows':len(tr),'training_last_year':int(tr.outcome_year.max()),'training_blocks':'|'.join(sorted(tr.block.unique()))})
sp=pd.DataFrame(spatial);sp.to_csv(OUT/'joint_block_time_predictions.csv',index=False)
ym=sp.groupby(['year','model']).AE.mean().unstack();ym['difference']=ym.burden-ym.baseline;ym.to_csv(OUT/'joint_block_time_year_metrics.csv')
bm=sp.groupby(['held_block','model']).AE.mean().unstack();bm['difference']=bm.burden-bm.baseline;bm.to_csv(OUT/'joint_block_time_block_metrics.csv')
sm=sp.groupby('model').AE.mean();spatial_summary={'n_units':int(len(sp)/2),'n_years':len(YEARS),'n_blocks':3,'pooled_MAE':sm.to_dict(),
 'year_equal_difference':float(ym.difference.mean()),'block3_95':block_interval(ym.difference),'better_years':int((ym.difference<0).sum()),'block_differences':bm.difference.to_dict()}
print('Spatial/time',json.dumps(spatial_summary),flush=True)

# Precompute per-fold abundance/weather predictions of log burdens.
folds=[]
for y in YEARS:
 tr=D[D.outcome_year<y].copy().reset_index(drop=True);te=D[D.outcome_year.eq(y)].copy()
 xb=design(tr);target=np.column_stack([np.log1p(tr.burden_m),np.log1p(tr.burden_c)])
 fit=solve(xb,xb,target);res=target-fit
 signatures={}
 for yr,g in tr.groupby('outcome_year'):
  signature=tuple(g.Plot);signatures.setdefault(signature,[]).append(g.index.to_numpy())
 folds.append((y,tr,te,fit,res,list(signatures.values())))
rng=np.random.default_rng(2026101301);null_rows=[];negative_clip=0
for rep in range(500):
 errs=[]
 for y,tr,te,fit,res,groups in folds:
  perm=res.copy()
  for buckets in groups:
   order=rng.permutation(len(buckets))
   for targetidx,srcidx in zip(buckets,order):perm[targetidx]=res[buckets[srcidx]]
  burden=np.expm1(fit+perm);negative_clip+=int((burden<0).sum());burden=np.maximum(burden,0)
  fake=tr.copy();fake['burden_m']=burden[:,0];fake['burden_c']=burden[:,1]
  vv=predict(fake,te,True);ae=abs(vv-te.outcome.to_numpy());errs.append(float(ae.mean()))
  null_rows.append({'replicate':rep,'year':y,'MAE':float(ae.mean())})
 if (rep+1)%100==0:print('Placebo',rep+1,flush=True)
nr=pd.DataFrame(null_rows);nr.to_csv(OUT/'residual_year_block_placebo_year_metrics.csv',index=False)
nd=nr.groupby('replicate').MAE.mean();nd.to_csv(OUT/'residual_year_block_placebo_distribution.csv')
actual=float(reference[reference.model.eq('fed_proxy')].groupby('outcome_year').absolute_error.mean().mean())
placebo={'real_year_equal_MAE':actual,'placebo_median_MAE':float(nd.median()),'placebo_2p5_97p5':np.quantile(nd,[.025,.975]).tolist(),
 'placebo_5pct':float(np.quantile(nd,.05)),'fraction_no_worse_than_real':float((nd<=actual).mean()),'repetitions':500,
 'clipped_pseudo_burden_cells':negative_clip,'inferential_label':'Fraction of the 500 residual-year controls whose MAE is no greater than the observed MAE.'}
summary={'original_max_abs_prediction_difference':maxdiff,'spatial_time':spatial_summary,'residual_placebo':placebo}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
