'Compare nymph-density forecasts with and without Daymet weather predictors.'
from pathlib import Path
import hashlib, json, math, platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/paper1_environment_benchmark_finite_v1_2026-10-16'
OLD=ROOT/'manuscript/paper1/ecology_letters_review_response_2026-10-14/diagnostics_reproducible_bundle/inputs'
sources=[OLD/'units.csv',OLD/'old_predictions.csv',Path(__file__),OUT/'seasonal_environment.csv',OUT/'ENVIRONMENT_QA.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file()}
u=pd.read_csv(OLD/'units.csv');old=pd.read_csv(OLD/'old_predictions.csv')
s=pd.read_csv(OUT/'seasonal_environment.csv')
d=u.merge(s,on='source_year',validate='many_to_one')
assert len(d)==174 and d.complete_92.all() and d.outcome_year.max()==2024
assert (d.outcome_year==d.source_year+1).all()
d.to_csv(OUT/'analysis_units_174.csv',index=False)
names=['original_base','original_burden','environment_base','environment_burden']
def features(q,name,plots):
    a=[np.log1p(q.mouse.to_numpy()),np.log1p(q.chip.to_numpy()),
       np.log1p(450*q.larvae.to_numpy()),q.past_gdd.to_numpy()]
    terms=['log1p_mouse','log1p_chip','log1p_larvae450','source_GDD']
    if name.endswith('burden'):
        a.extend([np.log1p(q.fed_m.to_numpy()),np.log1p(q.fed_c.to_numpy())]);terms+=['log1p_Fmouse','log1p_Fchip']
    a.extend((q.Plot==p).to_numpy(float) for p in plots[1:]);terms+=['grid_'+p for p in plots[1:]]
    if name.startswith('environment'):
        a.extend([q.summer_vpd_kPa.to_numpy(),q.summer_precip_mm.to_numpy()]);terms+=['summer_VPD_kPa','summer_precip_mm']
    return np.column_stack(a),terms
pred=[];scales=[];coefs=[]
# Check the baseline forecasts before adding weather predictors.
for name in names:
    for year in range(2002,2025):
        tr=d[d.outcome_year<year];te=d[d.outcome_year==year]
        assert tr.outcome_year.max()<year
        plots=sorted(tr.Plot.unique());a,terms=features(tr,name,plots);z,_=features(te,name,plots)
        mu=a.mean(axis=0);sd=a.std(axis=0,ddof=0);sd[sd<1e-10]=1
        x=np.column_stack([np.ones(len(tr)),(a-mu)/sd]);t=np.column_stack([np.ones(len(te)),(z-mu)/sd])
        penalty=np.eye(x.shape[1]);penalty[0,0]=0
        beta=np.linalg.solve(x.T@x+penalty,x.T@np.log1p(tr.outcome.to_numpy()))
        predicted=np.maximum(np.expm1(t@beta),0)
        for row,p in zip(te.itertuples(),predicted):
            pred.append(dict(model=name,outcome_year=year,source_year=row.source_year,Plot=row.Plot,
                             observed=row.outcome,predicted=float(p),absolute_error=abs(p-row.outcome),
                             train_rows=len(tr),train_last_outcome_year=int(tr.outcome_year.max())))
        scales.extend(dict(model=name,outcome_year=year,term=term,training_mean=float(m),training_sd=float(v)) for term,m,v in zip(terms,mu,sd))
        coefs.extend(dict(model=name,outcome_year=year,term=term,coefficient=float(v)) for term,v in zip(['intercept']+terms,beta))
    if name.startswith('original'):
        p=pd.DataFrame(pred);p=p[p.model.eq(name)]
        check=p.merge(old[old.model.eq('host_only' if name=='original_base' else 'fed_proxy')],on=['outcome_year','Plot'],suffixes=('','_old'),validate='one_to_one')
        assert len(check)==136
        assert abs(check.predicted-check.predicted_old).max()<1e-7
p=pd.DataFrame(pred);assert p.groupby('model').size().eq(136).all()
assert not p.duplicated(['model','outcome_year','Plot']).any()
p.to_csv(OUT/'forward_predictions_544.csv',index=False)
pd.DataFrame(scales).to_csv(OUT/'training_scaling.csv',index=False)
pd.DataFrame(coefs).to_csv(OUT/'training_coefficients.csv',index=False)
annual=p.groupby(['model','outcome_year']).absolute_error.mean().unstack('model').reindex(columns=names)
annual['original_delta']=annual.original_burden-annual.original_base
annual['environment_delta']=annual.environment_burden-annual.environment_base
annual.to_csv(OUT/'annual_errors_23.csv')
metrics=[]
for name in names:
    r=p[p.model.eq(name)]
    metrics.append(dict(model=name,n_units=len(r),n_years=23,pooled_MAE=float(r.absolute_error.mean()),
                        year_equal_MAE=float(annual[name].mean())))
pd.DataFrame(metrics).to_csv(OUT/'model_metrics.csv',index=False)
delta=annual.environment_delta.to_numpy();n=len(delta);est=[]
for length in [3,5]:
    rng=np.random.default_rng(2026101603)
    starts=rng.integers(0,n-length+1,size=(10000,math.ceil(n/length)))
    ix=(starts[:,:,None]+np.arange(length)).reshape(10000,-1)[:,:n]
    draws=delta[ix].mean(axis=1)
    np.savez_compressed(OUT/f'paired_bootstrap_L{length}.npz',calendar_years=annual.index.to_numpy(),indices=ix,delta_draws=draws)
    lo,hi=np.quantile(draws,[.025,.975])
    est.append(dict(comparison='environment_burden_minus_environment_base',block_length=length,primary=length==3,
                    estimate=float(delta.mean()),lower95=float(lo),upper95=float(hi),
                    improved_years=int((delta<0).sum()),n_years=n))
pd.DataFrame(est).to_csv(OUT/'primary_comparison.csv',index=False)
background={}
for label,base,new in [('original','original_base','original_burden'),('environment','environment_base','environment_burden')]:
    bm=next(x for x in metrics if x['model']==base);nm=next(x for x in metrics if x['model']==new)
    background[label]=dict(pooled_reduction=1-nm['pooled_MAE']/bm['pooled_MAE'],
                           year_equal_reduction=1-nm['year_equal_MAE']/bm['year_equal_MAE'],
                           year_equal_delta=float((annual[new]-annual[base]).mean()))
assert all(sha(ROOT/path)==v for path,v in before.items())
summary=dict(models=metrics,main_comparison=est,descriptive_reductions=background,
             all_174_units_preserved=True,all_136_test_units_preserved=True,original_frozen_predictions_reproduced=True,
             cutoff='source year December 31; data-release latency not modeled',
             retrospective_diagnostic_not_independent_confirmation=True,python=platform.python_version(),
             numpy=np.__version__,pandas=pd.__version__,new_model_specifications=2,new_forward_fits=46,
             original_reproduction_fits=46,old_files_unchanged=len(before),
             new_model_search=False,branch_completed=True)
(OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'FROZEN_INPUT_SHA256.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
