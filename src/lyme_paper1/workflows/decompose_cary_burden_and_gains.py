"""Two-part burden decomposition and paired temporal-block forecast comparisons."""
from pathlib import Path
import sys,json,warnings
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.analysis-deps'))
import statsmodels.api as sm
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
OUT=ROOT/'results/cary_model_extensions_2026-10-10'
d=pd.read_csv(ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv')
b=pd.read_csv(ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv')
gm={'Green Control':'GC','Green Experimental':'GE','Henry Control':'HC','Henry Experimental':'HE','Tea Control':'TC','Tea Experimental':'TE'}
b['Plot']=b.Grid.map(gm);b['doy']=pd.to_datetime(b.Day1).dt.dayofyear
b=b.merge(d[['source_year','Plot','mouse','chip','larvae','past_gdd']],left_on=['Year','Plot'],right_on=['source_year','Plot'],validate='many_to_one')
rows=[];meta=[]
for spp,g in b.groupby('Spp'):
    day=(g.doy-244)/30
    X=pd.DataFrame({'intercept':1.,'day':day,'day_sq':day**2,'supply':np.log1p(450*g.larvae),'gdd':(g.past_gdd-d.past_gdd.mean())/d.past_gdd.std(),'mouse':np.log1p(g.mouse),'chip':np.log1p(g.chip)})
    for p in sorted(d.Plot.unique())[1:]:X['plot_'+p]=(g.Plot==p).astype(float)
    print('two-part',spp,flush=True)
    fit=sm.GLM((g.Larvae>0).astype(int),X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g.Year})
    fits=[('occurrence_logit',fit)]
    pos=g.Larvae>0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        initial=sm.NegativeBinomial(g.Larvae,X,loglike_method='nb2').fit(disp=0,maxiter=300).params
        positive=TruncatedLFNegativeBinomialP(g.loc[pos,'Larvae'],X.loc[pos],truncation=0,p=2).fit(start_params=initial,method='lbfgs',bounds=[(None,None)]*X.shape[1]+[(1e-6,100.)],disp=0,maxiter=800,pgtol=1e-7,cov_type='cluster',cov_kwds={'groups':g.loc[pos,'Year']})
    fits.append(('positive_truncated_NB',positive))
    for label,f in fits:
        for term,val,se in zip(f.params.index,f.params,f.bse):rows.append({'species':spp,'component':label,'term':term,'estimate':val,'year_cluster_SE':se,'lower95':val-1.96*se,'upper95':val+1.96*se})
        meta.append({'species':spp,'component':label,'n':int(f.nobs),'AIC':float(f.aic),'converged':bool(getattr(f,'converged',getattr(f,'mle_retvals',{}).get('converged',False)))})
pd.DataFrame(rows).to_csv(OUT/'two_part_coefficients.csv',index=False);pd.DataFrame(meta).to_csv(OUT/'two_part_models.csv',index=False)
p=pd.read_csv(OUT/'chronological_predictions.csv');p['cohort']='long_peak_DON'
old=pd.read_csv(ROOT/'results/cary_compensation_forecast_2026-10-10/forward_predictions.csv');old=old[old.cohort=='long_peak_DON']
p=pd.concat([p,old],ignore_index=True);p['MSE_log']=p.log_error**2
y=p.groupby(['model','outcome_year']).agg(MAE=('absolute_error','mean'),MSE_log=('MSE_log','mean')).reset_index();y.to_csv(OUT/'combined_year_metrics.csv',index=False)
rng=np.random.default_rng(20261011);contrasts=[]
for base,model in [('host_only','fed_proxy'),('host_only','ANN_nested'),('ANN_nested_base','ANN_nested'),('fed_proxy','ANN_nested'),('mixed_base','mixed_burden'),('hetero_base','hetero_burden'),('mixed_burden','hetero_burden')]:
    a=y[y.model==base].set_index('outcome_year');c=y[y.model==model].set_index('outcome_year')
    for metric in ['MAE','MSE_log']:
        dv=(c[metric]-a[metric]).to_numpy();n=len(dv);draw=[]
        for _ in range(10000):
            starts=rng.integers(0,n-2,int(np.ceil(n/3)));idx=np.concatenate([np.arange(s,s+3) for s in starts])[:n];draw.append(float(dv[idx].mean()))
        contrasts.append({'baseline':base,'model':model,'metric':metric,'year_equal_difference':float(dv.mean()),'moving3year_block95':np.quantile(draw,[.025,.975]).tolist(),'better_years':int((dv<0).sum()),'n_years':n})
(OUT/'paired_gain_intervals.json').write_text(json.dumps(contrasts,indent=2))
print(pd.DataFrame(rows).query("term in ['mouse','chip']").to_string(index=False));print(json.dumps(contrasts,indent=2))
