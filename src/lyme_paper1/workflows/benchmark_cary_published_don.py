'Forecast nymph density using the model structure of Ostfeld et al. (2026).\n\nGaussian maximum likelihood integrates year, block and plot intercepts. Forecasts\nintegrate the new-year effect and condition on known spatial effects.\n'
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/published_benchmark_2026-10-11/don';OUT.mkdir(parents=True,exist_ok=True)
L=ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv'
R=ROOT/'data/raw/Full_dataset_20260326.csv'
raw=pd.read_csv(R)
d=pd.read_csv(L).merge(raw[['Year','Plot','RedOak_t2','PL_t1','cum_gdd0_oct_sep_t']],
 left_on=['outcome_year','Plot'],right_on=['Year','Plot'],suffixes=('','_target'),validate='one_to_one')
d=d.dropna(subset=['RedOak_t2','PL_t1','cum_gdd0_oct_sep_t']).copy()
d['block']=d.Plot.str[0];d['observed']=d.outcome;d.to_csv(OUT/'shared_units.csv',index=False)

def raw_features(q,model,stage):
    weather='past_gdd' if stage=='previous_year' else 'cum_gdd0_oct_sep_t'
    if model.startswith('Ostfeld'):
        cols=['RedOak_t2','PL_t1',weather]
        a=q[cols].to_numpy(float)
        if model.endswith('_burden'):
            a=np.column_stack([a,np.log1p(q.fed_m),np.log1p(q.fed_c)])
    else:
        a=np.column_stack([np.log1p(q.mouse),np.log1p(q.chip),np.log1p(450*q.larvae),q.past_gdd])
        if stage!='previous_year':a=np.column_stack([a,q.cum_gdd0_oct_sep_t])
        if model=='our_burden_ridge':a=np.column_stack([a,np.log1p(q.fed_m),np.log1p(q.fed_c)])
    return a

def fit_predict(tr,te,model,stage):
    a=raw_features(tr,model,stage);z=raw_features(te,model,stage)
    mu=a.mean(0);sd=np.where(a.std(0)>1e-10,a.std(0),1)
    X=np.column_stack([np.ones(len(tr)),(a-mu)/sd]);Z=np.column_stack([np.ones(len(te)),(z-mu)/sd])
    if not model.startswith('Ostfeld'):
        X=np.column_stack([X]+[(tr.Plot==p).to_numpy(float) for p in sorted(d.Plot.unique())[1:]])
        Z=np.column_stack([Z]+[(te.Plot==p).to_numpy(float) for p in sorted(d.Plot.unique())[1:]])
        P=np.eye(X.shape[1]);P[0,0]=0
        b=np.linalg.solve(X.T@X+P,X.T@np.log1p(tr.observed));res=np.log1p(tr.observed)-X@b
        median=np.maximum(np.expm1(Z@b),0);mean=np.maximum(np.expm1(Z@b+.5*np.var(res)),0)
        return median,mean,{'converged':True,'kind':'ridge_log1p','sigma2':float(np.var(res))}
    y=np.log(tr.observed.to_numpy())
    kernels=[(tr[c].to_numpy()[:,None]==tr[c].to_numpy()[None,:]).astype(float) for c in ['outcome_year','block','Plot']]
    spatial=[(te[c].to_numpy()[:,None]==tr[c].to_numpy()[None,:]).astype(float) for c in ['block','Plot']]
    initvar=max(float(np.var(y-X@np.linalg.lstsq(X,y,rcond=None)[0])),.01)
    def calc(v,keep=False):
        variance=np.exp(v);V=variance[0]*np.eye(len(tr))
        for a,K in zip(variance[1:],kernels):V+=a*K
        cf=cho_factor(V,lower=True,check_finite=False);viX=cho_solve(cf,X,check_finite=False);viy=cho_solve(cf,y,check_finite=False)
        b=np.linalg.solve(X.T@viX+np.eye(X.shape[1])*1e-10,X.T@viy);res=y-X@b
        nll=.5*(len(y)*np.log(2*np.pi)+2*np.log(np.diag(cf[0])).sum()+res@cho_solve(cf,res,check_finite=False))
        return (nll,b,variance,cf,res) if keep else nll
    fits=[minimize(calc,np.log(initvar*np.array(s)),method='L-BFGS-B',bounds=[(-16,8)]*4,
                   options={'maxiter':400,'ftol':1e-11}) for s in [[.6,.3,.05,.05],[.3,.4,.15,.15]]]
    f=min(fits,key=lambda r:r.fun);_,b,v,cf,res=calc(f.x,True)
    cross=v[2]*spatial[0]+v[3]*spatial[1]
    loc=Z@b+cross@cho_solve(cf,res,check_finite=False)
    # Population variation for a new observation includes an independent new year.
    predvar=np.maximum(v.sum()-np.sum(cross*cho_solve(cf,cross.T,check_finite=False).T,axis=1),1e-12)
    return np.exp(loc),np.exp(loc+.5*predvar),{'converged':bool(f.success),'kind':'Gaussian_log_random_intercepts',
            'sigma2':float(v[0]),'year_variance':float(v[1]),'block_variance':float(v[2]),'plot_variance':float(v[3]),'message':str(f.message)}

def weather_prediction(hist,yy,kind):
    if kind=='carry':return float(hist.iloc[-1].cum_gdd0_oct_sep_t)
    if kind=='mean10':return float(hist.tail(10).cum_gdd0_oct_sep_t.mean())
    h=hist.tail(10) if kind=='trend10' else hist
    x=h.Year.to_numpy()-h.Year.mean();b=np.linalg.lstsq(np.column_stack([np.ones(len(h)),x]),h.cum_gdd0_oct_sep_t,rcond=None)[0]
    return float(b[0]+b[1]*(yy-h.Year.mean()))

def forecast_weather(yy):
    h=raw[raw.Year<yy][['Year','cum_gdd0_oct_sep_t']].drop_duplicates().dropna().sort_values('Year')
    candidates=[]
    for kind in ['carry','mean10','trend10','trend_all']:
        errors=[]
        for v in h.Year.unique()[-5:]:
            past=h[h.Year<v]
            if len(past)<3:continue
            errors.append((weather_prediction(past,v,kind)-h[h.Year==v].cum_gdd0_oct_sep_t.iloc[0])**2)
        candidates.append((float(np.mean(errors)),kind))
    score,kind=min(candidates);return weather_prediction(h,yy,kind),kind,score

rows=[];diag=[];weather_rows=[]
for yy in sorted(d.outcome_year.unique()):
    if yy<2002:continue
    tr=d[d.outcome_year<yy];te=d[d.outcome_year==yy]
    wp,wmethod,wloss=forecast_weather(yy);weather_rows.append({'year':int(yy),'weather_prediction':wp,'method':wmethod,'inner_MSE':wloss})
    for stage in ['previous_year','previous_year_weather_forecast','after_weather']:
        target=te.copy()
        if stage=='previous_year_weather_forecast':target['cum_gdd0_oct_sep_t']=wp
        for model in ['Ostfeld2026_selected','Ostfeld2026_selected_burden','our_ecological_ridge','our_burden_ridge']:
            med,mean,info=fit_predict(tr,target,model,stage)
            diag.append({'year':int(yy),'stage':stage,'model':model,'train_n':len(tr),'train_last_year':int(tr.outcome_year.max()),**info})
            for point,vals in [('median',med),('mean',mean)]:
                for row,p in zip(te.to_dict('records'),vals):rows.append({'outcome_year':int(yy),'Plot':row['Plot'],'stage':stage,'model':model,'point':point,'observed':row['observed'],'predicted':float(p)})
    print('DON',yy,flush=True)
p=pd.DataFrame(rows);p['AE']=abs(p.predicted-p.observed);p['SE']=(p.predicted-p.observed)**2
p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(diag).to_csv(OUT/'fit_diagnostics.csv',index=False)
pd.DataFrame(weather_rows).to_csv(OUT/'weather_forecasts.csv',index=False)
metrics=p.groupby(['stage','model','point']).agg(n=('AE','size'),n_years=('outcome_year','nunique'),MAE=('AE','mean'),MSE=('SE','mean')).reset_index();metrics['RMSE']=np.sqrt(metrics.pop('MSE'));metrics.to_csv(OUT/'metrics.csv',index=False)
y=p.groupby(['stage','model','point','outcome_year']).AE.mean().reset_index();y.to_csv(OUT/'year_metrics.csv',index=False)
ym=y.groupby(['stage','model','point']).AE.mean().rename('year_equal_MAE').reset_index();metrics=metrics.merge(ym,on=['stage','model','point']);metrics.to_csv(OUT/'metrics.csv',index=False)
pairs=[];rng=np.random.default_rng(20261031)
for stage in p.stage.unique():
 for point in ['median','mean']:
  tab=y[(y.stage==stage)&(y.point==point)].pivot(index='outcome_year',columns='model',values='AE')
  for new,base in [('our_burden_ridge','Ostfeld2026_selected'),('our_burden_ridge','our_ecological_ridge'),('Ostfeld2026_selected_burden','Ostfeld2026_selected')]:
   dv=(tab[new]-tab[base]).to_numpy();n=len(dv);starts=rng.integers(0,n,(10000,int(np.ceil(n/3))));ix=((starts[:,:,None]+np.arange(3))%n).reshape(10000,-1)[:,:n];lo,hi=np.quantile(dv[ix].mean(1),[.025,.975])
   pairs.append({'stage':stage,'point':point,'new':new,'baseline':base,'n_years':n,'delta_MAE':dv.mean(),'relative_reduction':-dv.mean()/tab[base].mean(),'better_years':int((dv<0).sum()),'block95_lo':lo,'block95_hi':hi})
pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps({'python':platform.python_version(),'source_reimplementation':'Ostfeld et al. (2026) model structure: Gaussian likelihood for log density with year and spatial effects. The ridge comparison uses log1p nymph counts.',
 'stages':{'previous_year':'2026 weather covariate replaced by source-year GDD; lagged-feature sensitivity','previous_year_weather_forecast':'past observed target-weather in training, forecast target-weather in unknown year; weather forecast selected on past-only weather validation','after_weather':'Predict conditional on observed target-year cumulative GDD.'},
 'training':'all shared complete rows strictly before target year','point_outputs':'Report the conditional median and integrated observation mean.',
 'inputs':[{'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in [L,R]],'results':metrics.to_dict('records')},ensure_ascii=False,indent=2))
print(metrics.to_string(index=False),flush=True)
