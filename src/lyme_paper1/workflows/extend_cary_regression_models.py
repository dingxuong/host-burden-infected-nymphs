"""Individual NB burden, heterogeneous Gaussian mixed paths, chronological ANN.

Raw observation means reproduce the source data's capture-weighted estimand.
Year clusters are the inferential units for individual burden regression.
"""
from pathlib import Path
import sys, json, warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.analysis-deps'))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln, digamma
from scipy.stats import chi2, norm
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore',category=RuntimeWarning)
OUT=ROOT/'results/cary_model_extensions_2026-10-10';OUT.mkdir(exist_ok=True,parents=True)
d=pd.read_csv(ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv')
GM={'Green Control':'GC','Green Experimental':'GE','Henry Control':'HC','Henry Experimental':'HE','Tea Control':'TC','Tea Experimental':'TE'}
b=pd.read_csv(ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv')
b['Plot']=b.Grid.map(GM);b['doy']=pd.to_datetime(b.Day1).dt.dayofyear
b=b.merge(d[['source_year','Plot','mouse','chip','larvae','past_gdd']],left_on=['Year','Plot'],right_on=['source_year','Plot'],validate='many_to_one')

def nb_design(g,hosts=True):
    s=(g.doy.to_numpy()-244)/30
    x={'intercept':np.ones(len(g)),'day':s,'day_sq':s*s,'supply':np.log1p(450*g.larvae),'gdd':(g.past_gdd-d.past_gdd.mean())/d.past_gdd.std()}
    if hosts:x.update({'mouse':np.log1p(g.mouse),'chip':np.log1p(g.chip)})
    x.update({'plot_'+p:(g.Plot==p).astype(float) for p in sorted(d.Plot.unique())[1:]})
    return pd.DataFrame(x,index=g.index).astype(float)

nbrows=[];nbmeta=[];agg=[]
for spp,g in b.groupby('Spp'):
    y=g.Larvae.to_numpy();print('NB',spp,len(g),flush=True)
    for label,hosts in [('season_supply',False),('competition',True)]:
        X=nb_design(g,hosts)
        fit=sm.NegativeBinomial(y,X,loglike_method='nb2').fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':g.Year})
        for name,coef,se in zip(fit.params.index,fit.params,fit.bse):
            nbrows.append({'species':spp,'model':label,'term':name,'estimate':coef,'year_cluster_SE':se,'lower95':coef-1.96*se,'upper95':coef+1.96*se})
        nbmeta.append({'species':spp,'model':label,'n_observations':len(g),'n_years':g.Year.nunique(),'AIC':fit.aic,'converged':bool(fit.mle_retvals['converged']),'alpha':fit.params['alpha']})
    # NB2 dispersion depends on host abundance; k=1/alpha.
    # This is a likelihood extension; cluster intervals above belong to the homogeneous fits.
    X=nb_design(g).to_numpy();z=np.column_stack([np.ones(len(g)),(np.log1p(g.mouse)-np.log1p(d.mouse).mean())/np.log1p(d.mouse).std()])
    start=np.r_[fit.params.iloc[:-1].to_numpy(),np.log(fit.params['alpha']),0.]
    def objective(v):
        eta=X@v[:X.shape[1]];mu=np.exp(np.clip(eta,-25,25));la=z@v[X.shape[1]:];alpha=np.exp(np.clip(la,-12,12));k=1/alpha
        ll=gammaln(y+k)-gammaln(k)-gammaln(y+1)+k*np.log(k)-(y+k)*np.log(k+mu)+y*eta
        score=(y-mu)/(1+alpha*mu)
        dk=digamma(y+k)-digamma(k)+np.log(k)+1-np.log(k+mu)-(y+k)/(k+mu)
        grad=-np.r_[X.T@score,z.T@(-k*dk)]
        return -ll.sum(),grad
    opt=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10})
    nbmeta.append({'species':spp,'model':'density_dispersion','n_observations':len(g),'n_years':g.Year.nunique(),'AIC':2*len(opt.x)+2*opt.fun,'converged':bool(opt.success),'alpha_at_mean_log_mouse':float(np.exp(opt.x[-2])),'log_alpha_per_SD_log_mouse':float(opt.x[-1]),'optimizer_message':str(opt.message)})
    for (year,plot),u in g.groupby(['Year','Plot']):
        vals=np.sort(u.Larvae.to_numpy())[::-1];n20=int(np.ceil(.2*len(vals)))
        agg.append({'species':spp,'source_year':year,'Plot':plot,'n_observations':len(vals),'zero_fraction':float((vals==0).mean()),'top20_observation_share':float(vals[:n20].sum()/vals.sum()) if vals.sum() else np.nan})
pd.DataFrame(nbrows).to_csv(OUT/'individual_NB_coefficients.csv',index=False)
pd.DataFrame(nbmeta).to_csv(OUT/'individual_NB_models.csv',index=False)
pd.DataFrame(agg).to_csv(OUT/'burden_aggregation.csv',index=False)

def vars_frame(data):
    out=pd.DataFrame(index=data.index)
    for name,col in [('M','mouse'),('C','chip'),('L','larvae'),('Bm','burden_m'),('Bc','burden_c'),('A','RedOak_t1')]:
        out[name]=np.log1p(data[col]*(450 if name=='L' else 1))
    out['G']=data.past_gdd;out['D']=np.log1p(data.outcome)
    return out

def design(data,names,center=None,scale=None):
    f=vars_frame(data)[names].to_numpy();center=f.mean(0) if center is None else center;scale=f.std(0) if scale is None else scale;scale=np.where(scale<1e-10,1,scale)
    X=np.column_stack([np.ones(len(data)),(f-center)/scale]+[(data.Plot==p).astype(float) for p in sorted(d.Plot.unique())[1:]])
    return X,center,scale

def gaussian_year_model(data,response,names,hetero=False):
    """Exact marginal Gaussian ML, year random intercept, plot fixed intercepts.
    Var(residual_i)=exp(g0+g1*z(GDD_i)) for mouse abundance, or
    exp(g0+g1*z(log1p mouse_i)) for burden/DON; year blocks integrated exactly.
    """
    X,center,scale=design(data,names);y=vars_frame(data)[response].to_numpy();years=data.outcome_year if response=='D' else data.source_year
    blocks=[np.where(years.to_numpy()==yr)[0] for yr in sorted(years.unique())]
    h=data.past_gdd.to_numpy() if response=='M' else np.log1p(data.mouse.to_numpy());hc=h.mean();hs=h.std();h=(h-hc)/hs
    init=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@init;v=max(res.var(),.01)
    start=np.r_[np.log(v*.8),np.log(v*.2),0.] if hetero else np.log([v*.8,v*.2])
    def calculation(q,keep=False):
        rvar=np.exp(np.clip(q[0]+(q[2]*h if hetero else 0),-20,20))*np.ones(len(y));tau=np.exp(q[1]);sx=np.zeros((X.shape[1],X.shape[1]));sy=np.zeros(X.shape[1]);logdet=0.;blocks_v=[]
        for idx in blocks:
            w=1/rvar[idx];den=1/tau+w.sum();xx=X[idx];yy=y[idx]
            vx=w[:,None]*xx-np.outer(w,w@xx)/den;vy=w*yy-w*(w@yy)/den
            sx+=xx.T@vx;sy+=xx.T@vy;logdet+=np.log(rvar[idx]).sum()+np.log1p(tau*w.sum());blocks_v.append((idx,w,den))
        coef=np.linalg.solve(sx+np.eye(len(sx))*1e-10,sy);r=y-X@coef;quad=0.
        for idx,w,den in blocks_v:quad+=(w*r[idx]**2).sum()-(w@r[idx])**2/den
        nll=.5*(len(y)*np.log(2*np.pi)+logdet+quad)
        if keep:return coef,np.linalg.inv(sx),rvar,tau,nll
        return nll
    bounds=[(-15,15),(-15,15)]+([(-5,5)] if hetero else [])
    fit=minimize(calculation,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':300,'ftol':1e-11})
    coef,cov,rvar,tau,nll=calculation(fit.x,True)
    return {'coef':coef,'se':np.sqrt(np.diag(cov)),'center':center,'scale':scale,'nll':float(nll),'AIC':float(2*nll+2*(len(coef)+len(fit.x))),'year_variance':float(tau),'residual_variance_at_mean':float(np.exp(fit.x[0])),'variance_slope':float(fit.x[2]) if hetero else 0.,'variance_covariate':'GDD' if response=='M' else 'log1p_mouse','success':bool(fit.success),'optimizer':str(fit.message),'names':['intercept']+names+['plot_'+p for p in sorted(d.Plot.unique())[1:]]}

paths=d.dropna(subset=['RedOak_t1']).copy();prows=[];pmeta=[];pfits={}
components={'M':['A','G','L','C'],'Bm':['M','C','L','G'],'Bc':['M','C','L','G'],'D':['M','C','L','G','Bm','Bc']}
for response,names in components.items():
    print('path',response,flush=True)
    for hetero in [False,True]:
        f=gaussian_year_model(paths,response,names,hetero);label='heterogeneous' if hetero else 'homogeneous'
        pfits[response,label]=f
        pmeta.append({'response':response,'model':label,'n':len(paths),**{k:v for k,v in f.items() if k in ['AIC','year_variance','residual_variance_at_mean','variance_slope','variance_covariate','success','optimizer']}})
        for name,coef,se in zip(f['names'],f['coef'],f['se']):
            prows.append({'response':response,'model':label,'term':name,'estimate_per_SD_predictor':coef,'SE_conditional_on_variances':se,'lower95':coef-1.96*se,'upper95':coef+1.96*se})
pd.DataFrame(prows).to_csv(OUT/'mixed_path_coefficients.csv',index=False);pd.DataFrame(pmeta).to_csv(OUT/'mixed_path_models.csv',index=False)
# Three acorn conditional-independence implications of the stated DAG.
# Correlated Bm/Bc residuals and exogenous node covariances are allowed.
ds=[]
for response in ['Bm','Bc','D']:
    f=gaussian_year_model(paths,response,components[response]+['A'],True);i=f['names'].index('A');zval=f['coef'][i]/f['se'][i];pval=float(2*norm.sf(abs(zval)))
    ds.append({'independence':f'{response} independent of A given parents and year/plot','p_Wald':pval})
Cstat=float(-2*sum(np.log(max(v['p_Wald'],1e-300)) for v in ds))
sem={'n':len(paths),'year_random_plot_fixed':True,'basis_set':ds,'Fisher_C':Cstat,'df':6,'p':float(chi2.sf(Cstat,6)),'coefficient_interval':'Gaussian fixed-effect information conditional on estimated variance parameters','DAG':'A,G,L,C -> M; M,C,L,G -> Bm and Bc; M,C,L,G,Bm,Bc -> D; exogenous covariances and Bm/Bc residual covariance allowed','note':'Path products describe associations among log1p-transformed variables.'}
fm=pfits['Bm','heterogeneous'];fc=pfits['Bc','heterogeneous'];fd=pfits['D','heterogeneous']
sem['mouse_direct_D']=float(fd['coef'][fd['names'].index('M')]/fd['scale'][components['D'].index('M')])
for label,f in [('Bm',fm),('Bc',fc)]:
    a=f['coef'][f['names'].index('M')]/f['scale'][components[label].index('M')]
    bb=fd['coef'][fd['names'].index(label)]/fd['scale'][components['D'].index(label)]
    sem['mouse_indirect_via_'+label]=float(a*bb)
(OUT/'piecewise_paths.json').write_text(json.dumps(sem,indent=2))

pred=[];selection=[];ann_diagnostics=[]
names=['M','C','L','G','Bm','Bc']
def ann_predict(train,test,hidden,alpha,nn):
    xx,mu,sd=design(train,nn);zz,_,_=design(test,nn,mu,sd)
    # Initial column is an explicit intercept for regression; ANN supplies its own.
    xx=xx[:,1:];zz=zz[:,1:];y=np.log1p(train.outcome.to_numpy());ym=y.mean();ys=max(y.std(),1e-8);pp=[]
    for seed in [13,37,71]:
        net=MLPRegressor(hidden_layer_sizes=(hidden,),activation='tanh',solver='lbfgs',alpha=alpha,max_iter=1200,random_state=seed,tol=1e-6)
        with warnings.catch_warnings():warnings.simplefilter('ignore');net.fit(xx,(y-ym)/ys)
        ann_diagnostics.append({'n_predictors':xx.shape[1],'train_rows':len(train),'training_last_year':int(train.outcome_year.max()),'prediction_first_year':int(test.outcome_year.min()),'hidden_nodes':hidden,'alpha':alpha,'seed':seed,'iterations':net.n_iter_,'reached_iteration_limit':bool(net.n_iter_>=net.max_iter),'training_loss':float(net.loss_)})
        pp.append(net.predict(zz)*ys+ym)
    return np.mean(pp,axis=0)
years=sorted(d.outcome_year.unique())
for year in years[10:]:
    print('forecast',year,flush=True);tr=d[d.outcome_year<year];te=d[d.outcome_year==year]
    for burden in [False,True]:
        nn=names if burden else names[:4]
        for hetero in [False,True]:
            f=gaussian_year_model(tr,'D',nn,hetero);X,_,_=design(te,nn,f['center'],f['scale']);yp=X@f['coef'];label=('hetero' if hetero else 'mixed')+('_burden' if burden else '_base')
            for row,v in zip(te.to_dict('records'),yp):pred.append({'model':label,'outcome_year':year,'Plot':row['Plot'],'observed':row['outcome'],'predicted':float(max(np.expm1(v),0)),'log_error':float(v-np.log1p(row['outcome'])),'training_last_year':int(tr.outcome_year.max())})
    # Candidate selection is internal: last three observed training years.
    inner=sorted(tr.outcome_year.unique())[-3:]
    for burden in [False,True]:
        scores=[];nn=names if burden else names[:4];label='ANN_nested' if burden else 'ANN_nested_base'
        for hidden in [2,4]:
            for alpha in [1.,10.,100.]:
                losses=[]
                for vy in inner:
                    it=tr[tr.outcome_year<vy];iv=tr[tr.outcome_year==vy]
                    pv=ann_predict(it,iv,hidden,alpha,nn);losses.append(float(np.mean((pv-np.log1p(iv.outcome))**2)))
                scores.append((np.mean(losses),hidden,alpha))
        best=sorted(scores)[0];selection.append({'model':label,'test_year':year,'hidden_nodes':best[1],'alpha':best[2],'inner_year_equal_MSE':best[0],'candidate_scores':str(scores)})
        yp=ann_predict(tr,te,best[1],best[2],nn)
        for row,v in zip(te.to_dict('records'),yp):pred.append({'model':label,'outcome_year':year,'Plot':row['Plot'],'observed':row['outcome'],'predicted':float(max(np.expm1(v),0)),'log_error':float(v-np.log1p(row['outcome'])),'training_last_year':int(tr.outcome_year.max())})
p=pd.DataFrame(pred);p['absolute_error']=(p.predicted-p.observed).abs();p.to_csv(OUT/'chronological_predictions.csv',index=False)
pd.DataFrame(selection).to_csv(OUT/'ANN_nested_selection.csv',index=False)
pd.DataFrame(ann_diagnostics).to_csv(OUT/'ANN_optimizer_diagnostics.csv',index=False)
metrics=p.groupby('model').agg(n=('observed','size'),MAE=('absolute_error','mean'),log_RMSE=('log_error',lambda x:np.sqrt(np.mean(x**2)))).reset_index();metrics.to_csv(OUT/'chronological_metrics.csv',index=False)
run={'individual_rows':len(b),'raw_rows':35885,'complete_gridyears':len(d),'path_gridyears':len(paths),'excluded_individual_rule':'only source gridyears with complete published host/supply/GDD and matched next-year DON','individual_estimand':'capture observation count; repeated individuals retained; cluster uncertainty by year','ANN':'single tanh hidden layer; mean of 3 fixed seeds; nodes 2/4 and alpha 1/10/100 selected using last 3 training years forward validation, test year excluded','forecast':'The same 136 test plot-years as the ridge comparison; train on earlier years, set the new-year effect to zero and report log-scale medians.','metrics':metrics.to_dict('records'),'NB_models':nbmeta,'paths':sem}
(OUT/'summary.json').write_text(json.dumps(run,indent=2));print(json.dumps(run,indent=2))
