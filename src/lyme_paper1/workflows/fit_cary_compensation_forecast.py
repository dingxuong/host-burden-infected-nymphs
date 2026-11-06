'Forecast Cary nymph density from measurements in the preceding host year.\n'
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
parser=argparse.ArgumentParser()
parser.add_argument('--alpha',type=float,default=1.)
parser.add_argument('--tag-average',action='store_true')
parser.add_argument('--acorns',action='store_true')
parser.add_argument('--window-years',type=int,default=0)
parser.add_argument('--variant',default='')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/cary_compensation_forecast_2026-10-10'
if args.variant:OUT=OUT/args.variant
OUT.mkdir(exist_ok=True,parents=True)
GM={'Green Control':'GC','Green Experimental':'GE','Henry Control':'HC','Henry Experimental':'HE','Tea Control':'TC','Tea Experimental':'TE'}
b=pd.read_csv(ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv')
b['date']=pd.to_datetime(b.Day1); b['Plot']=b.Grid.map(GM)
if args.tag_average:
 b=b[b.Tag.notna()].groupby(['Year','Plot','Spp','Tag'],as_index=False).Larvae.mean()
ag=b.groupby(['Year','Plot','Spp']).Larvae.agg(['mean','count']).unstack('Spp');ag.columns=['_'.join(c) for c in ag.columns];ag=ag.reset_index()
l=pd.read_csv(ROOT/'data/raw/Full_dataset_20260326.csv');l=l[l.Plot.notna()].copy()
s=l[['Year','Plot','PL_t','TS_t','larval_peak_t','cum_gdd0_oct_sep_t','RedOak_t1','OtherOak_t1']].merge(ag,on=['Year','Plot'],validate='one_to_one')
s=s.rename(columns={'Year':'source_year','PL_t':'mouse','TS_t':'chip','larval_peak_t':'larvae','cum_gdd0_oct_sep_t':'past_gdd','mean_PL':'burden_m','mean_TS':'burden_c'})
s['outcome_year']=s.source_year+1
j=s.merge(l[['Year','Plot','nymph_peak_t']],left_on=['outcome_year','Plot'],right_on=['Year','Plot'],validate='one_to_one')
j=j.dropna(subset=['mouse','chip','larvae','burden_m','burden_c','past_gdd','nymph_peak_t']).copy()
if args.acorns:j=j.dropna(subset=['RedOak_t1','OtherOak_t1']).copy()
j['fed_m']=j.mouse*j.burden_m;j['fed_c']=j.chip*j.burden_c
j['outcome']=j.nymph_peak_t*450
j['cohort']='long_peak_DON';j.to_csv(OUT/'long_units.csv',index=False)
c=pd.read_csv(ROOT/'data/raw/cary_2025/allthedata_grid.csv');c['Plot']=c.Grid.map({'greenC':'GC','greenE':'GE','henryC':'HC','henryE':'HE','teaC':'TC','teaE':'TE'})
c['source_year']=c.Year-1;c['outcome_year']=c.Year
for dest,src in [('mouse','PL.mnRawLyr'),('chip','TS.mnRawLyr'),('burden_m','PLlarv.mn'),('burden_c','TSlarv.mn')]:c[dest]=c[src]
c['fed_m']=c.mouse*c.burden_m;c['fed_c']=c.chip*c.burden_c
# Omit larval supply where its measurement year cannot be established.
c['cohort']='short_mean_DIN';c['outcome']=450*c.mnNm2*c.Bburgd
ass=pd.read_csv(ROOT/'data/raw/cary_2025/NymphalInfectionData.csv').groupby(['Year','Grid'])['B.burgdorferi'].agg(['sum','count']).reset_index()
c=c.merge(ass,on=['Year','Grid'],validate='one_to_one');c.to_csv(OUT/'infection_units.csv',index=False)
def features(d,model,knot,plots,long):
 z=[np.log1p(d.mouse.to_numpy()),np.log1p(d.chip.to_numpy())]
 if long:z.extend([np.log1p(450*d.larvae.to_numpy()),d.past_gdd.to_numpy()])
 if long and args.acorns:z.extend([np.log1p(d.RedOak_t1.to_numpy()),np.log1p(d.OtherOak_t1.to_numpy())])
 if model=='fed_proxy':z.extend([np.log1p(d.fed_m.to_numpy()),np.log1p(d.fed_c.to_numpy())])
 elif model=='separate_burden':z.extend([np.log1p(d.burden_m.to_numpy()),np.log1p(d.burden_c.to_numpy())])
 elif model=='nonlinear_compensation':
  x=np.log1p(d.mouse.to_numpy());z.extend([np.log1p(d.fed_m.to_numpy()),np.log1p(d.fed_c.to_numpy()),np.maximum(x-knot,0)])
 z.extend([(d.Plot==p).to_numpy().astype(float) for p in plots[1:]])
 return np.column_stack(z)
def fitpredict(train,test,model,long):
 plots=sorted(train.Plot.unique());knot=float(np.median(np.log1p(train.mouse)))
 X=features(train,model,knot,plots,long);Z=features(test,model,knot,plots,long)
 mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
 X=np.column_stack([np.ones(len(X)),(X-mu)/sd]);Z=np.column_stack([np.ones(len(Z)),(Z-mu)/sd]); y=np.log1p(train.outcome.to_numpy())
 P=np.eye(X.shape[1]);P[0,0]=0
 coef=np.linalg.solve(X.T@X+args.alpha*P,X.T@y);pred_log=Z@coef
 return np.maximum(np.expm1(pred_log),0), pred_log, knot
modern_don=c.copy();modern_don['outcome']=450*modern_don.mnNm2;modern_don['cohort']='short_mean_DON'
matched_peak=c.merge(l[['Year','Plot','nymph_peak_t']],on=['Year','Plot'],validate='one_to_one')
matched_peak['outcome']=450*matched_peak.nymph_peak_t;matched_peak['cohort']='matched_short_peak_DON'
pred=[]
for d,long,min_years in [(j,True,10),(c,False,4),(modern_don,False,4),(matched_peak,False,4)]:
 years=sorted(d.outcome_year.unique())
 for year in years[min_years:]:
  tr=d[d.outcome_year<year];te=d[d.outcome_year==year]
  if args.window_years:tr=tr[tr.outcome_year>=year-args.window_years]
  for model in ['host_only','fed_proxy','separate_burden','nonlinear_compensation']:
   yp,zp,knot=fitpredict(tr,te,model,long)
   for idx,(row,pv,zv) in enumerate(zip(te.to_dict('records'),yp,zp)):
    pred.append({'cohort':row['cohort'],'outcome_year':int(year),'Plot':row['Plot'],'model':model,'observed':row['outcome'],'predicted':float(pv),'log_error':float(zv-np.log1p(row['outcome'])),'absolute_error':float(abs(pv-row['outcome'])),'training_rows':len(tr),'training_last_outcome_year':int(tr.outcome_year.max()),'host_knot':float(np.expm1(knot))})
p=pd.DataFrame(pred);p.to_csv(OUT/'forward_predictions.csv',index=False)
met=[]
for (cohort,model),g in p.groupby(['cohort','model']):
 met.append({'cohort':cohort,'model':model,'n_units':len(g),'n_years':g.outcome_year.nunique(),'first_test_year':int(g.outcome_year.min()),'last_test_year':int(g.outcome_year.max()),'MAE_per450m2':float(g.absolute_error.mean()),'RMSE_log1p_per450m2':float(np.sqrt((g.log_error**2).mean()))})
m=pd.DataFrame(met);m.to_csv(OUT/'forward_metrics.csv',index=False)
y=p.groupby(['cohort','outcome_year','model']).agg(MAE=('absolute_error','mean'),MSE_log=('log_error',lambda z:np.mean(z**2))).reset_index();y.to_csv(OUT/'year_metrics.csv',index=False)
rng=np.random.default_rng(20261010);deltas=[]
for cohort,g in y.groupby('cohort'):
 base=g[g.model=='host_only'].set_index('outcome_year')
 for model in ['fed_proxy','separate_burden','nonlinear_compensation']:
  x=g[g.model==model].set_index('outcome_year');dv=x.MAE-base.MAE;idx=rng.integers(0,len(dv),(10000,len(dv)));draw=dv.to_numpy()[idx].mean(1)
  deltas.append({'cohort':cohort,'model':model,'delta_MAE':float(dv.mean()),'better_years':int((dv<0).sum()),'test_years':len(dv),'year_resampling_95pct':np.quantile(draw,[.025,.975]).tolist()})
# Conditional compensation curve. Same-year abundance contrasts after plot, year and larval supply adjustment.
curves=[]
for controls in ['pooled','plot_supply','plot_year_supply']:
 X=[np.ones(len(j)),np.log(j.mouse.to_numpy())]
 if controls!='pooled':
  X.append(np.log1p(450*j.larvae.to_numpy()))
  X.extend(pd.get_dummies(j.Plot,drop_first=True,dtype=float).to_numpy().T)
 if controls=='plot_year_supply':X.extend(pd.get_dummies(j.source_year.astype(str),drop_first=True,dtype=float).to_numpy().T)
 X=np.column_stack(X);coef=np.linalg.lstsq(X,np.log(j.burden_m),rcond=None)[0]
 curves.append({'adjustment':controls,'individual_burden_log_elasticity':float(coef[1]),'fed_proxy_log_elasticity':float(1+coef[1])})
summary={'long_n_units':len(j),'long_source_year_range':[int(j.source_year.min()),int(j.source_year.max())],'long_raw_burden_observations':int(j[['count_PL','count_TS']].sum().sum()),'compensation_elasticities':curves,'forward_metrics':met,'paired_year_deltas':deltas,'specification':{'ridge_penalty':args.alpha,'window_years':args.window_years,'include_acorns_tminus2':args.acorns,'standardization':'training only','nonlinear_knot':'training median mouse abundance','long_baseline':'plot + past mouse/chip abundance + source-year questing larvae and source-year degree days' + (' + oak acorns two years before outcome' if args.acorns else ''),'short_baseline':'plot + previous-year mouse/chip MNA','all_evaluation':('rolling '+str(args.window_years)+' years' if args.window_years else 'expanding window') + ', all observations of target year withheld','forecast_units':'nymphs or infected nymphs per450m2','short_outcome':'seasonal mean DON times Bb prevalence','long_outcome':'peak DON','burden_summary':'mean of identified-tag annual means' if args.tag_average else 'Mean burden across captures in each source year, including repeated captures of the same host.','deployment_information':'source-year measurements only; actual data-release dates not modeled'}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
