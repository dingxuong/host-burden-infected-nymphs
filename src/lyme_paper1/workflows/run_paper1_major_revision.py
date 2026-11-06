'Compare host-abundance estimates, burden summaries and forecasting assumptions.'
from pathlib import Path
import ast, hashlib, json, platform
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper1_major_revision_2026-10-14'
OUT.mkdir(exist_ok=True)
INPUT = ROOT/'results/cary_compensation_forecast_2026-10-10/long_units.csv'
REF = ROOT/'results/cary_compensation_forecast_2026-10-10/forward_predictions.csv'
RAW = ROOT/'data/raw/Full_dataset_20260326.csv'
HOST = ROOT/'data/raw/cary_related/host.TrapData_1991-2022.csv'
BURDEN = ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv'
MODERN = ROOT/'results/cary_transmission_paths_2026-10-10/modern_units.csv'
SCOT = ROOT/'manuscript/paper1/submission_cn_2026-10-13/external_scale_results/all_72_observational_units.csv'
ANN = ROOT/'results/cary_model_extensions_2026-10-10/ANN_nested_selection.csv'
WEIGHT = ROOT/'data/raw/cary_related/lagged_data_20251014.csv'
paths=[INPUT, REF, RAW, HOST, BURDEN, MODERN, SCOT, ANN, WEIGHT]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p):sha(p)for p in paths}
D=pd.read_csv(INPUT).sort_values(['outcome_year','Plot']).reset_index(drop=True)
PLOTS=sorted(D.Plot.unique()); YEARS=list(range(2002,2025))
BLOCK={'GC':'Green','GE':'Green','HC':'Henry','HE':'Henry','TC':'Tea','TE':'Tea'}
GM={'Green Control':'GC','Green Experimental':'GE','Henry Control':'HC','Henry Experimental':'HE','Tea Control':'TC','Tea Experimental':'TE'}
D['block']=D.Plot.map(BLOCK)
for col in ['fed_m','fed_c']:
    z=np.log1p(D[col]);D[col+'_between']=z.groupby(D.source_year).transform('mean')
    D[col+'_within']=z-D[col+'_between']

def design(d, model):
    cols=[np.log1p(450*d.larvae.to_numpy()),d.past_gdd.to_numpy()]
    if 'H' in model:cols += [np.log1p(d.mouse),np.log1p(d.chip)]
    if 'B' in model:cols += [np.log1p(d.burden_m),np.log1p(d.burden_c)]
    if 'F' in model:cols += [np.log1p(d.mouse*d.burden_m),np.log1p(d.chip*d.burden_c)]
    if 'between' in model:cols += [d.fed_m_between,d.fed_c_between]
    if 'within' in model:cols += [d.fed_m_within,d.fed_c_within]
    if 'JS' in model:cols += [np.log1p(d[x])for x in ['js_full_m','js_full_c','js_het_m','js_het_c']]
    if 'weight' in model:cols += [d.AvgMeanWt]
    cols += [(d.Plot==p).astype(float)for p in PLOTS[1:]]
    return np.column_stack(cols)

def prediction(tr,te,model,alpha=1.):
    X=design(tr,model);Z=design(te,model)
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
    X=np.column_stack([np.ones(len(X)),(X-mu)/sd]);Z=np.column_stack([np.ones(len(Z)),(Z-mu)/sd])
    penalty=np.eye(X.shape[1]);penalty[0,0]=0
    beta=np.linalg.solve(X.T@X+alpha*penalty,X.T@np.log1p(tr.outcome))
    logp=Z@beta
    return np.maximum(np.expm1(logp),0),logp

def forward(data,models,alphas=(1.,),excluded=(),nested=False):
    rows=[];tune=[]
    for y in YEARS:
        if y in excluded:continue
        tr=data[(data.outcome_year<y)&~data.outcome_year.isin(excluded)]
        te=data[data.outcome_year==y]
        if te.empty:continue
        if tr.outcome_year.nunique()<10:raise RuntimeError(('Training history',y))
        for model in models:
            values=alphas
            if nested:
                score=[]
                for alpha in alphas:
                    losses=[]
                    for yy in sorted(tr.outcome_year.unique())[-5:]:
                        it=tr[tr.outcome_year<yy];iv=tr[tr.outcome_year==yy]
                        _,lp=prediction(it,iv,model,alpha)
                        losses.append(np.mean((lp-np.log1p(iv.outcome))**2))
                    score.append((float(np.mean(losses)),float(alpha)))
                chosen=min(score)[1];values=[chosen]
                tune.extend({'year':y,'model':model,'alpha':a,'inner_log_MSE':s,'selected':a==chosen}for s,a in score)
            for alpha in values:
                pp,lp=prediction(tr,te,model,alpha)
                for r,v,z in zip(te.to_dict('records'),pp,lp):
                    rows.append({'year':y,'Plot':r['Plot'],'block':r['block'],'model':model,'alpha':alpha,
                                 'observed':r['outcome'],'predicted':v,'AE':abs(v-r['outcome']),
                                 'log_error':z-np.log1p(r['outcome']), 'train_last_year':int(tr.outcome_year.max()),'train_rows':len(tr)})
    return pd.DataFrame(rows),pd.DataFrame(tune)

def interval(v,seed=2026101411):
    v=np.asarray(v);rng=np.random.default_rng(seed);n=len(v)
    starts=rng.integers(0,n,(10000,int(np.ceil(n/3))))
    ix=((starts[:,:,None]+np.arange(3))%n).reshape(10000,-1)[:,:n]
    return np.quantile(v[ix].mean(1),[.025,.975]).tolist()

def metrics(pred):
    result=[]
    for (model,a),g in pred.groupby(['model','alpha']):
        yr=g.groupby('year').AE.mean()
        result.append({'model':model,'alpha':a,'n_units':len(g),'n_years':len(yr),
                       'pooled_MAE':float(g.AE.mean()),'year_equal_MAE':float(yr.mean()),
                       'log_RMSE':float(np.sqrt(np.mean(g.log_error**2)))})
    return pd.DataFrame(result)

def contrast(pred,new,base,alpha=1.):
    t=pred[pred.alpha==alpha].groupby(['year','model']).AE.mean().unstack()
    v=t[new]-t[base]
    return {'new':new,'base':base,'year_difference':float(v.mean()),'block95':interval(v),
            'better_years':int((v<0).sum()),'n_years':len(v)}

models=['E','EH','EB','EHB','EF','EHF','EHBF','EH_between','EH_within','EH_between_within']
p,_=forward(D,models);p.to_csv(OUT/'representation_predictions.csv',index=False)
met=metrics(p);met.to_csv(OUT/'representation_metrics.csv',index=False)
ref=pd.read_csv(REF);ref=ref[(ref.cohort=='long_peak_DON')&ref.model.isin(['host_only','fed_proxy'])].copy()
ref['model']=ref.model.map({'host_only':'EH','fed_proxy':'EHF'})
repro=p[p.model.isin(['EH','EHF'])].merge(ref,on=['Plot','model'],suffixes=('','_ref'))
repro=repro[repro.year==repro.outcome_year]
maxdiff=float(abs(repro.predicted-repro.predicted_ref).max())
if maxdiff>1e-8 or len(repro)!=272:raise RuntimeError(('Predictions differ from the expected values',maxdiff,len(repro)))
summary={'representation':met.to_dict('records'),'contrasts':[contrast(p,x,'EH')for x in models if x!='EH'],
         'frozen_prediction_max_diff':maxdiff}
print('Representation',met.to_string(index=False),flush=True)

# Alternative observed abundance estimates, common complete cohort.
h=pd.read_csv(HOST);h['date']=pd.to_datetime(h.Date,format='%m/%d/%y');h=h[h.date.dt.month.isin([8,9])].copy()
h['Plot']=h.Grid.map(GM)
h=h.rename(columns={'Full.(birth.and.death)':'js_full','Heterogeneous.survival':'js_het'})
av=h.groupby(['Year','Plot','spec'])[['js_full','js_het']].mean().unstack('spec')
av.columns=[a+('_m' if b=='PL'else '_c')for a,b in av.columns];av=av.reset_index().rename(columns={'Year':'source_year'})
js=D.merge(av,on=['source_year','Plot'],how='left',validate='one_to_one')
js=js.dropna(subset=['js_full_m','js_full_c','js_het_m','js_het_c']).copy()
js.to_csv(OUT/'all_JS_complete_units.csv',index=False)
rows=[];jsc=[]
for kind in ['original','js_full','js_het','original_plus_JS']:
    dd=js.copy();variants=['EH','EHF']
    if kind in ['js_full','js_het']:dd['mouse']=dd[kind+'_m'];dd['chip']=dd[kind+'_c']
    if kind=='original_plus_JS':variants=['EH_JS','EHF_JS']
    pp,_=forward(dd,variants);pp['abundance_definition']=kind;rows.append(pp)
    tab=metrics(pp);tab['abundance_definition']=kind
    for r in tab.to_dict('records'):jsc.append(r)
    cc=contrast(pp,variants[1],variants[0]);cc['abundance_definition']=kind
    summary.setdefault('JS_contrasts',[]).append(cc)
pd.concat(rows).to_csv(OUT/'JS_abundance_predictions.csv',index=False)
pd.DataFrame(jsc).to_csv(OUT/'JS_abundance_metrics.csv',index=False);summary['JS_metrics']=jsc
print('JS',json.dumps(summary['JS_contrasts']),flush=True)

# Every leave-one-year deletion is refitted, with fixed joint exclusions.
influence=[]
for label,ex in [(f'leave_{y}',[y])for y in YEARS]+[('outcome_2016_2017',[2016,2017]),('source_2016_2017',[2017,2018])]:
    pp,_=forward(D,['EH','EHF'],excluded=ex);cc=contrast(pp,'EHF','EH');cc['deletion']=label;cc['excluded_outcome_years']=ex
    cc['year_MAE_baseline']=float(pp[pp.model=='EH'].groupby('year').AE.mean().mean())
    cc['year_MAE_burden']=float(pp[pp.model=='EHF'].groupby('year').AE.mean().mean())
    influence.append(cc)
pd.DataFrame(influence).to_csv(OUT/'refitted_year_influence.csv',index=False);summary['influence']=influence

# Regularization and nested past-only choice.
pen,_=forward(D,['EH','EHF'],alphas=[.1,1.,10.,100.]);pen.to_csv(OUT/'ridge_penalty_predictions.csv',index=False)
metrics(pen).to_csv(OUT/'ridge_penalty_metrics.csv',index=False)
summary['penalty_contrasts']=[dict(alpha=a,**contrast(pen,'EHF','EH',a))for a in [.1,1.,10.,100.]]
nested,tune=forward(D,['EH','EHF'],alphas=[.1,1.,10.,100.],nested=True)
nested.to_csv(OUT/'ridge_nested_predictions.csv',index=False);tune.to_csv(OUT/'ridge_nested_tuning.csv',index=False)
npred=nested.copy();npred.alpha=1.;summary['nested_ridge']=contrast(npred,'EHF','EH');summary['nested_ridge_metrics']=metrics(npred).to_dict('records')
condition=[]
for y in YEARS:
    tr=D[D.outcome_year<y];X=design(tr,'EHF');sd=X.std(0);active=sd>1e-10
    Z=(X[:,active]-X[:,active].mean(0))/sd[active]
    singular=np.linalg.svd(Z,compute_uv=False)
    row={'year':y,'condition_number':float(singular.max()/singular.min()),'max_VIF':0.}
    for j in range(Z.shape[1]):
        z=Z[:,j];other=np.delete(Z,j,axis=1);b=np.linalg.lstsq(other,z,rcond=None)[0]
        resid=np.sum((z-other@b)**2);vif=float((z@z)/resid)if resid>1e-12 else float('inf')
        row['max_VIF']=max(row['max_VIF'],vif)
        if j in [2,3,4,5]:row[f'VIF_feature_{j}']=vif
    condition.append(row)
pd.DataFrame(condition).to_csv(OUT/'training_condition_diagnostics.csv',index=False)

# Full grid-level distribution and time-by-spatial-block sensitivity.
units=p[p.model.isin(['EH','EHF'])].pivot(index=['year','Plot','block'],columns='model',values='AE').reset_index()
units['difference']=units.EHF-units.EH;units.to_csv(OUT/'all_136_paired_grid_errors.csv',index=False)
annual=units.groupby('year').difference.agg(['mean','median','min','max','count',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
annual.columns=['mean','median','min','max','n_grids','q25','q75'];annual.to_csv(OUT/'all_23_year_grid_distributions.csv')
rng=np.random.default_rng(2026101413);sums=units.groupby(['year','block']).difference.sum().unstack().fillna(0).to_numpy()
counts=units.groupby(['year','block']).difference.count().unstack().fillna(0).to_numpy()
starts=rng.integers(0,23,(10000,8));ix=((starts[:,:,None]+np.arange(3))%23).reshape(10000,-1)[:,:23]
bl=rng.integers(0,3,(10000,3));values=[]
for yi,bi in zip(ix,bl):
    ss=sums[yi][:,bi].sum(1);nn=counts[yi][:,bi].sum(1);valid=nn>0
    values.append(float(np.mean(ss[valid]/nn[valid])))
pd.DataFrame({'draw':np.arange(10000),'year_equal_difference':values}).to_csv(OUT/'spatiotemporal_resample_draws.csv',index=False)
summary['crossed_resampling']={'point':float(annual['mean'].mean()),'interval95':np.quantile(values,[.025,.975]).tolist(),
                              'unit':'three complete spatial-block trajectories crossed with three-year blocks'}
summary['sign_count_descriptive']={'better':int((annual['mean']<0).sum()),'years':len(annual),
                                 'iid_sign_p':float(binomtest(int((annual['mean']<0).sum()),len(annual),.5,alternative='greater').pvalue),
                                 'role':'Descriptive iid-year reference; temporal-block interval remains primary.'}

# Missing cells: source predictors versus target DON, original rows retained.
raw=pd.read_csv(RAW);bb=pd.read_csv(BURDEN);bb['Plot']=bb.Grid.map(GM)
missing=[]
for y in YEARS:
    for plot in PLOTS:
        if not ((D.outcome_year==y)&(D.Plot==plot)).any():
            prev=raw[(raw.Year==y-1)&(raw.Plot==plot)];nex=raw[(raw.Year==y)&(raw.Plot==plot)]
            g=bb[(bb.Year==y-1)&(bb.Plot==plot)]
            missing.append({'outcome_year':y,'source_year':y-1,'Plot':plot,
                            'source_missing_fields':[x for x in ['PL_t','TS_t','larval_peak_t','cum_gdd0_oct_sep_t']if prev[x].isna().any()],
                            'burden_species_available':sorted(g.Spp.unique()),'outcome_DON_present':bool(nex.nymph_peak_t.notna().any())})
summary['missing_cells']=missing
summary['individual_available_columns']=pd.read_csv(BURDEN,nrows=0).columns.tolist()

# Compare the neural-network configurations.
sel=pd.read_csv(ANN);candidates=[]
for r in sel.to_dict('records'):
    for mse,n,a in ast.literal_eval(r['candidate_scores']):
        candidates.append({'model':r['model'],'year':r['test_year'],'hidden':n,'alpha':a,'inner_MSE':mse,
                           'selected':n==r['hidden_nodes']and a==r['alpha']})
pd.DataFrame(candidates).to_csv(OUT/'ANN_all_276_candidate_scores.csv',index=False)
sel.to_csv(OUT/'ANN_original_selected_hyperparameters.csv',index=False)

# Annual mouse mass is a separate grid-year covariate, never an individual trait.
wt=pd.read_csv(WEIGHT).rename(columns={'Year':'source_year','Grid':'Plot'})
wt=D.merge(wt[['source_year','Plot','AvgMeanWt']],on=['source_year','Plot'],how='inner',validate='one_to_one')
wp,_=forward(wt,['EH_weight','EHF_weight']);wp.to_csv(OUT/'annual_weight_predictions.csv',index=False)
summary['annual_weight']={'n_complete_units':len(wt),'metrics':metrics(wp).to_dict('records'),
                          'contrast':contrast(wp,'EHF_weight','EH_weight'),'covariate':'Mean mouse mass in August for each plot-year.'}

def elasticity(d,year_effect=False):
    cols=[np.ones(len(d)),np.log(d.mouse),np.log1p(450*d.larvae)]
    cols.extend(pd.get_dummies(d.Plot,drop_first=True,dtype=float).to_numpy().T)
    if year_effect:cols.extend(pd.get_dummies(d.source_year,drop_first=True,dtype=float).to_numpy().T)
    return float(np.linalg.lstsq(np.column_stack(cols),np.log(d.burden_m),rcond=None)[0][1])
summary['elasticity_without_source_2016_2017']={str(y):elasticity(D[~D.source_year.isin([2016,2017])],y)for y in [False,True]}
summary['collinearity']={'condition_min':min(x['condition_number']for x in condition),'condition_max':max(x['condition_number']for x in condition),
                         'max_VIF':max(x['max_VIF']for x in condition)}

# Compare ecological-year scales at fixed DON.
c=pd.read_csv(MODERN).sort_values(['outcome_year','Plot']);all_years=sorted(c.outcome_year.unique())
q=np.log(c.pivot(index='outcome_year',columns='Plot',values='DON_mean').to_numpy())
lp=np.log(c.pivot(index='outcome_year',columns='Plot',values='NIP').to_numpy())
tq=np.diff(q,axis=0);tp=np.diff(lp,axis=0);sq=q-q.mean(1,keepdims=True);sp=lp-lp.mean(1,keepdims=True)
def fraction(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float((np.var(b)+np.mean((a-a.mean())*(b-b.mean())))/np.var(a+b))
scale_points={'full_spatial':fraction(sq,sp),'temporal':fraction(tq,tp),'matched_spatial':fraction(sq[1:],sp[1:])}
scale_points['full_difference']=scale_points['full_spatial']-scale_points['temporal']
scale_points['matched_difference']=scale_points['matched_spatial']-scale_points['temporal']
rng=np.random.default_rng(2026101412);scale=[]
for method in ['iid_year','circular_two_year']:
    for rep in range(10000):
        ix=rng.integers(0,8,8)if method=='iid_year'else ((rng.integers(0,8,4)[:,None]+np.arange(2))%8).ravel()
        st=fraction(tq[ix],tp[ix]);ss=fraction(sq[1:][ix],sp[1:][ix])
        full_s=fraction(np.vstack([sq[0],sq[1:][ix]]),np.vstack([sp[0],sp[1:][ix]]))
        scale.append({'method':method,'draw':rep,'temporal':st,'matched_spatial':ss,'matched_difference':ss-st,'full_spatial_difference':full_s-st})
scale=pd.DataFrame(scale);scale.to_csv(OUT/'paired_ecological_scale_draws.csv',index=False)
summary['ecological_scale']={'points':scale_points,'resampling':[],'matched_end_years':all_years[1:]}
for method,g in scale.groupby('method'):
    summary['ecological_scale']['resampling'].append({'method':method,'matched_difference95':np.quantile(g.matched_difference,[.025,.975]).tolist(),
                                                    'full_difference95':np.quantile(g.full_spatial_difference,[.025,.975]).tolist(),
                                                    'matched_fraction_positive':float((g.matched_difference>0).mean())})

# Separate Jeffreys smoothing, random risk ratios, and reference-set redefinition.
c=pd.read_csv(SCOT);rng=np.random.default_rng(2026101304);scot=[]
for (yr,visit),g in c.groupby(['Year','Visit'],sort=False):
    g=g.sort_values('Site_code');density=g.DON_100.to_numpy();ids=g.Site_code.to_numpy();m=len(g)//3
    chosen=np.lexsort((ids,-density))[:m];rawrisk=density*g.k.to_numpy()/g.n.to_numpy()
    rawbest=np.lexsort((ids,-rawrisk))[:m];meanrisk=density*(g.k.to_numpy()+.5)/(g.n.to_numpy()+1)
    meanbest=np.lexsort((ids,-meanrisk))[:m]
    pp=rng.beta(g.k.to_numpy()+.5,g.n.to_numpy()-g.k.to_numpy()+.5,(20000,len(g)));risk=pp*density
    adaptive=np.sort(risk,axis=1)[:,-m:].sum(1)
    adaptive_ratio=risk[:,chosen].sum(1)/adaptive
    fixed_ratio=risk[:,chosen].sum(1)/risk[:,rawbest].sum(1)
    scot.append({'year':int(yr),'visit':visit,'quota':m,'raw_ratio':float(rawrisk[chosen].sum()/rawrisk[rawbest].sum()),
                 'ratio_of_expected_risks_fixed_original':float(meanrisk[chosen].sum()/meanrisk[rawbest].sum()),
                 'smoothed_mean_rank_ratio':float(meanrisk[chosen].sum()/meanrisk[meanbest].sum()),
                 'mean_random_ratio_fixed_original_set':float(fixed_ratio.mean()),
                 'mean_random_ratio_reoptimized_set':float(adaptive_ratio.mean()),
                 'reoptimized_minus_fixed_mean':float(np.mean(adaptive_ratio-fixed_ratio)),
                 'fixed_ratio_above_one_fraction':float((fixed_ratio>1).mean()),
                 'PCR_n_min':int(g.n.min()),'PCR_n_max':int(g.n.max()),'zero_positive_sites':int((g.k==0).sum())})
pd.DataFrame(scot).to_csv(OUT/'Scottish_ratio_estimand_decomposition.csv',index=False)
summary['Scotland_decomposition']=scot;summary['Scotland_raw_equal_visit_mean']=float(np.mean([r['raw_ratio']for r in scot]))
summary['Scotland_conditional_equal_visit_mean']=float(np.mean([r['mean_random_ratio_reoptimized_set']for r in scot]))
summary['permutation_empirical_plus_one_position']=1/501
summary['branch_status']='Host-abundance, burden-representation and forecasting sensitivity analyses.'
if not all(sha(Path(f))==h for f,h in hashes.items()):raise RuntimeError('Input changed')
(OUT/'RUN_MANIFEST.json').write_text(json.dumps({'inputs':hashes,'python':platform.python_version(),
                                              'script_sha256':sha(Path(__file__)),'no_future_training':bool((p.train_last_year<p.year).all())},indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({k:summary[k]for k in ['penalty_contrasts','nested_ridge','crossed_resampling','ecological_scale','Scotland_raw_equal_visit_mean','Scotland_conditional_equal_visit_mean','annual_weight']},ensure_ascii=False,indent=2,default=lambda x:x.item()),flush=True)
