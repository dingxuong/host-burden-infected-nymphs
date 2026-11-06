'Compare weather forecasts and neural-network validation windows.'
from pathlib import Path
import ast,json,hashlib,warnings,time,sys
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-deps'))
from sklearn.neural_network import MLPRegressor

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper1_major_revision_2026-10-14'
OUT.mkdir(exist_ok=True)
shared=ROOT/'results/published_benchmark_2026-10-11/don/shared_units.csv'
rawpath=ROOT/'data/raw/Full_dataset_20260326.csv';Dpath=ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv'
source=ROOT/'scripts/benchmark_cary_published_don.py'
oldann=ROOT/'results/cary_model_extensions_2026-10-10/chronological_predictions.csv'
oldsel=ROOT/'results/cary_model_extensions_2026-10-10/ANN_nested_selection.csv'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
inputs={str(f):sha(f)for f in [shared,rawpath,Dpath,source,oldann,oldsel]}
threadpool_limits(1)
d=pd.read_csv(shared);raw=pd.read_csv(rawpath)
# Load the model definitions.
tree=ast.parse(source.read_text());funs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
ns={'np':np,'pd':pd,'cho_factor':cho_factor,'cho_solve':cho_solve,'minimize':minimize,'d':d,'raw':raw}
exec(compile(ast.Module(body=funs,type_ignores=[]),str(source),'exec'),ns)
weather=[];rows=[];diag=[];optim_cache={};current=None
def cached_minimize(fun,x,*args,**kwargs):
    key=(current,tuple(np.round(x,10)))
    if key not in optim_cache:optim_cache[key]=minimize(fun,x,*args,**kwargs)
    return optim_cache[key]
ns['minimize']=cached_minimize
for yy in range(2002,2025):
    tr=d[d.outcome_year<yy];te=d[d.outcome_year==yy]
    hist=raw[raw.Year<yy][['Year','cum_gdd0_oct_sep_t']].drop_duplicates().dropna().sort_values('Year')
    for rule in ['carry','mean10']:
        wp=ns['weather_prediction'](hist,yy,rule)
        weather.append({'year':yy,'rule':rule,'predicted_gdd':wp,'last_weather_year':int(hist.Year.max())})
        target=te.copy();target['cum_gdd0_oct_sep_t']=wp
        for model in ['Ostfeld2026_selected','Ostfeld2026_selected_burden','our_ecological_ridge','our_burden_ridge']:
            current=(yy,model)
            med,_,info=ns['fit_predict'](tr,target,model,'previous_year_weather_forecast')
            diag.append({'year':yy,'rule':rule,'model':model,**info})
            for r,v in zip(te.to_dict('records'),med):
                rows.append({'year':yy,'Plot':r['Plot'],'rule':rule,'model':model,'observed':r['observed'],
                             'predicted':v,'AE':abs(v-r['observed']),'training_last_year':int(tr.outcome_year.max())})
    print('fixed weather',yy,flush=True)
weather=pd.DataFrame(weather);w=pd.DataFrame(rows);weather.to_csv(OUT/'fixed_weather_forecasts.csv',index=False)
w.to_csv(OUT/'fixed_weather_predictions.csv',index=False);pd.DataFrame(diag).to_csv(OUT/'fixed_weather_fit_diagnostics.csv',index=False)
def interval(v):
    v=np.asarray(v);n=len(v);rng=np.random.default_rng(2026101411)
    st=rng.integers(0,n,(10000,int(np.ceil(n/3))))
    ix=((st[:,:,None]+np.arange(3))%n).reshape(10000,-1)[:,:n]
    return np.quantile(v[ix].mean(1),[.025,.975]).tolist()
wm=[];wc=[]
for rule,g in w.groupby('rule'):
    y=g.groupby(['year','model']).AE.mean().unstack()
    for model,gg in g.groupby('model'):wm.append({'rule':rule,'model':model,'n_units':len(gg),'pooled_MAE':float(gg.AE.mean()),'year_equal_MAE':float(y[model].mean())})
    for new,base in [('Ostfeld2026_selected_burden','Ostfeld2026_selected'),('our_burden_ridge','our_ecological_ridge')]:
        v=y[new]-y[base];wc.append({'rule':rule,'new':new,'baseline':base,'difference':float(v.mean()),'block95':interval(v),'better_years':int((v<0).sum()),'reduction':float(-v.mean()/y[base].mean())})
pd.DataFrame(wm).to_csv(OUT/'fixed_weather_metrics.csv',index=False)

# Same small ANN, same candidate set and seeds; five past years instead of three.
D=pd.read_csv(Dpath);plots=sorted(D.Plot.unique());cache={};ad=[]
original_ann_source=ROOT/'scripts/extend_cary_regression_models.py'
ann_ns={'np':np,'pd':pd,'d':D}
ann_tree=ast.parse(original_ann_source.read_text())
ann_functions=[n for n in ann_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['vars_frame','design']]
exec(compile(ast.Module(body=ann_functions,type_ignores=[]),str(original_ann_source),'exec'),ann_ns)
def raw_ann(t,burden):
    x=[np.log1p(t.mouse),np.log1p(t.chip),np.log1p(450*t.larvae),t.past_gdd]
    if burden:x +=[np.log1p(t.burden_m),np.log1p(t.burden_c)]
    return np.column_stack(x)
def ann(tr,te,burden,hidden,alpha):
    key=(int(tr.outcome_year.max()),burden,hidden,alpha)
    if key not in cache:
        names=['M','C','L','G','Bm','Bc']if burden else ['M','C','L','G']
        xx,mu,sd=ann_ns['design'](tr,names);xx=xx[:,1:]
        y=np.log1p(tr.outcome.to_numpy());ym=y.mean();ys=max(y.std(),1e-8);nets=[]
        for seed in [13,37,71]:
            net=MLPRegressor(hidden_layer_sizes=(hidden,),activation='tanh',solver='lbfgs',alpha=alpha,
                             max_iter=1200,random_state=seed,tol=1e-6)
            with warnings.catch_warnings():warnings.simplefilter('ignore');net.fit(xx,(y-ym)/ys)
            nets.append(net);ad.append({'training_last_year':key[0],'burden':burden,'hidden':hidden,'alpha':alpha,'seed':seed,'n_iter':net.n_iter_,
                                       'iteration_limit':net.n_iter_>=1200,'training_loss':float(net.loss_)})
        cache[key]=(mu,sd,ym,ys,nets)
    mu,sd,ym,ys,nets=cache[key]
    names=['M','C','L','G','Bm','Bc']if burden else ['M','C','L','G']
    xx,_,_=ann_ns['design'](te,names,mu,sd);xx=xx[:,1:]
    return np.mean([net.predict(xx)*ys+ym for net in nets],axis=0)
sel=[];pr=[];old=pd.read_csv(oldsel);repro=[]
for yy in range(2002,2025):
    tr=D[D.outcome_year<yy];te=D[D.outcome_year==yy]
    for burden in [False,True]:
        model='ANN_nested'if burden else 'ANN_nested_base'
        scores=[]
        for hidden in [2,4]:
            for a in [1.,10.,100.]:
                losses=[]
                for iy in sorted(tr.outcome_year.unique())[-5:]:
                    it=tr[tr.outcome_year<iy];iv=tr[tr.outcome_year==iy]
                    lp=ann(it,iv,burden,hidden,a);losses.append(np.mean((lp-np.log1p(iv.outcome))**2))
                scores.append((float(np.mean(losses)),hidden,a))
        best=min(scores)
        sel.extend({'year':yy,'model':model,'hidden':h,'alpha':a,'inner_log_MSE':s,'selected':h==best[1]and a==best[2]}for s,h,a in scores)
        lp=ann(tr,te,burden,best[1],best[2])
        for r,z in zip(te.to_dict('records'),lp):
            v=max(np.expm1(z),0);pr.append({'year':yy,'Plot':r['Plot'],'model':model,'observed':r['outcome'],'predicted':v,'AE':abs(v-r['outcome']),
                                          'hidden':best[1],'alpha':best[2],'train_last_year':int(tr.outcome_year.max())})
        rr=old[(old.test_year==yy)&(old.model==model)].iloc[0]
        zp=ann(tr,te,burden,int(rr.hidden_nodes),float(rr.alpha))
        repro.extend({'outcome_year':yy,'Plot':r['Plot'],'model':model,'reproduced':max(np.expm1(z),0)}for r,z in zip(te.to_dict('records'),zp))
    print('ANN five-year validation',yy,flush=True)
ap=pd.DataFrame(pr);ap.to_csv(OUT/'ANN_five_year_predictions.csv',index=False)
pd.DataFrame(sel).to_csv(OUT/'ANN_five_year_all_candidates.csv',index=False);pd.DataFrame(ad).to_csv(OUT/'ANN_five_year_optimizer_diagnostics.csv',index=False)
rp=pd.DataFrame(repro).merge(pd.read_csv(oldann)[['outcome_year','Plot','model','predicted']],on=['outcome_year','Plot','model'],validate='one_to_one')
maxdiff=float(abs(rp.reproduced-rp.predicted).max());rp.to_csv(OUT/'ANN_original_parameter_reproduction.csv',index=False)
if maxdiff>1e-5:raise RuntimeError(('ANN original predictions do not match',maxdiff))
y=ap.groupby(['year','model']).AE.mean().unstack();v=y.ANN_nested-y.ANN_nested_base
result={'weather_metrics':wm,'weather_contrasts':wc,'ANN_window5':{'metrics':[{ 'model':m,'pooled_MAE':float(g.AE.mean()),'year_equal_MAE':float(y[m].mean())}for m,g in ap.groupby('model')],
                                                                'paired_difference':float(v.mean()),'block95':interval(v),'better_years':int((v<0).sum()),'old_parameter_reproduction_maxdiff':maxdiff},
        'all_source_hashes_unchanged':all(sha(Path(f))==h for f,h in inputs.items()),'inputs':inputs,'script_sha256':sha(Path(__file__))}
(OUT/'weather_ANN_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
