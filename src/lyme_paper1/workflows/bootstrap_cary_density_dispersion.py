"""Year-cluster bootstrap for the NB2 density-dependent dispersion slope."""
from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.optimize import minimize
from scipy.special import gammaln,digamma
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/cary_model_extensions_2026-10-10'
d=pd.read_csv(ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv');g=pd.read_csv(ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv')
gm={'Green Control':'GC','Green Experimental':'GE','Henry Control':'HC','Henry Experimental':'HE','Tea Control':'TC','Tea Experimental':'TE'}
g['Plot']=g.Grid.map(gm);g['day']=(pd.to_datetime(g.Day1).dt.dayofyear-244)/30
g=g[g.Spp=='PL'].merge(d[['source_year','Plot','mouse','chip','larvae','past_gdd']],left_on=['Year','Plot'],right_on=['source_year','Plot'])
X=np.column_stack([np.ones(len(g)),g.day,g.day**2,np.log1p(450*g.larvae),(g.past_gdd-d.past_gdd.mean())/d.past_gdd.std(),np.log1p(g.mouse),np.log1p(g.chip)]+[(g.Plot==p).astype(float) for p in sorted(d.Plot.unique())[1:]])
y=g.Larvae.to_numpy();z=np.column_stack([np.ones(len(g)),(np.log1p(g.mouse)-np.log1p(d.mouse).mean())/np.log1p(d.mouse).std()])
co=pd.read_csv(OUT/'individual_NB_coefficients.csv');co=co[(co.species=='PL')&(co.model=='competition')];start=np.r_[co.estimate.iloc[:-1].to_numpy(),np.log(co.estimate.iloc[-1]),0.]
def fit(weights,start):
    def objective(v):
        eta=X@v[:X.shape[1]];mu=np.exp(np.clip(eta,-25,25));alpha=np.exp(np.clip(z@v[X.shape[1]:],-12,12));k=1/alpha
        ll=gammaln(y+k)-gammaln(k)-gammaln(y+1)+k*np.log(k)-(y+k)*np.log(k+mu)+y*eta
        score=(y-mu)/(1+alpha*mu);dk=digamma(y+k)-digamma(k)+np.log(k)+1-np.log(k+mu)-(y+k)/(k+mu)
        return -(weights*ll).sum(),-np.r_[X.T@(weights*score),z.T@(weights*(-k*dk))]
    return minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':350,'ftol':1e-9})
full=fit(np.ones(len(g)),start);years,idx=np.unique(g.Year,return_inverse=True);rng=np.random.default_rng(20261012);draw=[]
for i in range(400):
    counts=np.bincount(rng.integers(0,len(years),len(years)),minlength=len(years));r=fit(counts[idx],full.x)
    draw.append({'replicate':i,'slope':float(r.x[-1]),'converged':bool(r.success),'gradient_max':float(np.max(np.abs(r.jac)))})
    if i%50==0:print('bootstrap',i,flush=True)
pd.DataFrame(draw).to_csv(OUT/'density_dispersion_year_bootstrap.csv',index=False)
valid=[x['slope'] for x in draw if x['converged']];s={'species':'PL','n_years':len(years),'replicates':400,'converged':len(valid),'slope':float(full.x[-1]),'year_cluster_percentile95':np.quantile(valid,[.025,.975]).tolist(),'standardization':'log1p mouse abundance standardized using the 174 original gridyears; fixed in bootstrap','weights':'each sampled source year includes every capture observation and every plot in that year'}
(OUT/'density_dispersion_uncertainty.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
