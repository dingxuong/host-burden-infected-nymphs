"""Forecast model comparisons, risk-component checks and site-selection diagnostics."""
from pathlib import Path
import argparse,json,hashlib,platform,math
import numpy as np,pandas as pd
from scipy.stats import beta as beta_distribution
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--inputs',type=Path,required=True)
args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
FILES={k:args.inputs/(k+'.csv') for k in ['units','raw','old_predictions','scotland','modern','transitions','pcr']}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in FILES.items()}
D=pd.read_csv(FILES['units']).sort_values(['outcome_year','Plot']);raw=pd.read_csv(FILES['raw']);lag=raw[['Year','Plot','nymph_peak_t']].dropna(subset=['Plot']).rename(columns={'Year':'source_year','nymph_peak_t':'lag_DON_m2'});D=D.merge(lag,on=['source_year','Plot'],how='left',validate='one_to_one');D['lag_DON']=450*D.lag_DON_m2
assert len(D)==174 and D.lag_DON.notna().all();D.to_csv(OUT/'all_174_dynamic_units.csv',index=False)
PLOTS=sorted(D.Plot.unique());YEARS=list(range(2002,2025));tuning=[];basis_audit=[];fit_count=0

def continuous(d,kind):
 cols={}
 if kind not in ['AR','ARF']:cols.update(mouse=np.log1p(d.mouse),chip=np.log1p(d.chip),larvae=np.log1p(450*d.larvae),gdd=d.past_gdd)
 if 'AR' in kind:cols['lag_DON']=np.log1p(d.lag_DON)
 if kind.endswith('F'):cols.update(fed_m=np.log1p(d.fed_m),fed_c=np.log1p(d.fed_c))
 return cols

def matrices(tr,te,kind,spline=False):
 ct=continuous(tr,kind);ce=continuous(te,kind);X=[];Z=[];audit=[]
 for name,x in ct.items():
  x=np.asarray(x,dtype=float);z=np.asarray(ce[name],dtype=float);mu=x.mean();sd=x.std();sd=sd if sd>1e-10 else 1.;x=(x-mu)/sd;z=(z-mu)/sd
  X.append(x);Z.append(z)
  if spline:
   knots=np.unique(np.quantile(x,[.1,.3,.5,.7,.9]));audit.append({'feature':name,'raw_training_mean':mu,'raw_training_sd':sd,'knots_standardized':knots.tolist()})
   if len(knots)>=3:
    a,b=knots[-2:];scale=(b-knots[0])**2
    for k in knots[:-2]:
     basis=lambda u:(np.maximum(u-k,0)**3-(b-k)/(b-a)*np.maximum(u-a,0)**3+(a-k)/(b-a)*np.maximum(u-b,0)**3)/scale
     X.append(basis(x));Z.append(basis(z))
 for plot in PLOTS[1:]:X.append((tr.Plot==plot).astype(float));Z.append((te.Plot==plot).astype(float))
 return np.column_stack(X),np.column_stack(Z),audit

def predict(tr,te,kind,alpha=1.,spline=False):
 global fit_count
 X,Z,audit=matrices(tr,te,kind,spline);mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1.;X=np.column_stack([np.ones(len(X)),(X-mu)/sd]);Z=np.column_stack([np.ones(len(Z)),(Z-mu)/sd]);pen=np.eye(X.shape[1]);pen[0,0]=0
 b=np.linalg.solve(X.T@X+alpha*pen,X.T@np.log1p(tr.outcome));lp=Z@b;fit_count+=1
 return np.maximum(np.expm1(lp),0),lp,audit

rows=[]
for yr in YEARS:
 tr=D[D.outcome_year<yr];te=D[D.outcome_year==yr];assert tr.outcome_year.nunique()>=10
 for name,kind,spline in [('eco','E',False),('eco_F','EF',False),('AR','AR',False),('AR_F','ARF',False),('eco_AR','EAR',False),('eco_AR_F','EARF',False),('spline_eco','E',True),('spline_eco_F','EF',True),('spline_eco_AR','EAR',True),('spline_eco_AR_F','EARF',True),('persistence','P',False)]:
  alpha=1.
  if name=='persistence':pp=te.lag_DON.to_numpy();lp=np.log1p(pp);audit=[]
  else:
   if spline:
    scores=[]
    for a in [.1,1.,10.,100.]:
     loss=[]
     for vy in sorted(tr.outcome_year.unique())[-5:]:
      it=tr[tr.outcome_year<vy];iv=tr[tr.outcome_year==vy];p,_,_=predict(it,iv,kind,a,True);loss.append(float(np.mean(abs(iv.outcome.to_numpy()-p))))
     scores.append((float(np.mean(loss)),a))
    alpha=min(scores)[1]
    tuning.extend({'year':yr,'model':name,'alpha':a,'inner_year_equal_MAE':v,'selected':a==alpha} for v,a in scores)
   pp,lp,audit=predict(tr,te,kind,alpha,spline)
  if spline:basis_audit.append({'year':yr,'model':name,'features':audit})
  for r,p,logp in zip(te.to_dict('records'),pp,lp):rows.append({'year':yr,'source_year':r['source_year'],'Plot':r['Plot'],'model':name,'alpha':alpha,'observed':r['outcome'],'lag_DON':r['lag_DON'],'predicted':p,'absolute_error':abs(p-r['outcome']),'log_error':logp-np.log1p(r['outcome']),'train_last_year':int(tr.outcome_year.max()),'train_rows':len(tr)})
 print('Completed forecast year',yr,flush=True)
pred=pd.DataFrame(rows);pred.to_csv(OUT/'all_1496_forward_predictions.csv',index=False);pd.DataFrame(tuning).to_csv(OUT/'past_only_spline_selection.csv',index=False);(OUT/'spline_basis_audit.json').write_text(json.dumps(basis_audit,indent=2))
old=pd.read_csv(FILES['old_predictions']);old['new_model']=old.model.map({'host_only':'eco','fed_proxy':'eco_F'});check=old.merge(pred,left_on=['outcome_year','Plot','new_model'],right_on=['year','Plot','model'],validate='one_to_one',suffixes=('_old','_new'));md=float(abs(check.predicted_new-check.predicted_old).max());assert len(check)==272 and md<1e-8
met=[]
for name,g in pred.groupby('model'):
 annual=g.groupby('year').absolute_error.mean();met.append({'model':name,'n_units':len(g),'n_years':len(annual),'pooled_MAE':float(g.absolute_error.mean()),'year_equal_MAE':float(annual.mean()),'log_RMSE':float(np.sqrt(np.mean(g.log_error**2)))})
pd.DataFrame(met).to_csv(OUT/'model_metrics.csv',index=False)
annual=pred.groupby(['year','model']).absolute_error.mean().unstack();annual.to_csv(OUT/'all_23_annual_MAE.csv')
pairs=[('eco_F','eco'),('AR_F','AR'),('eco_AR_F','eco_AR'),('spline_eco_F','spline_eco'),('spline_eco_AR_F','spline_eco_AR')]
intervals=[];draw_rows=[];contrasts=[]
for new,base in pairs:
 v=(annual[new]-annual[base]).to_numpy();contrasts.append({'new':new,'base':base,'year_equal_difference':float(v.mean()),'better_years':int((v<0).sum()),'n_years':len(v)})
 for length in [1,3,5,8]:
  rng=np.random.default_rng(2026101431);n=len(v);starts=rng.integers(0,n-length+1,(10000,math.ceil(n/length)));ix=(starts[:,:,None]+np.arange(length)).reshape(10000,-1)[:,:n];bs=v[ix].mean(1);lo,hi=np.quantile(bs,[.025,.975]);intervals.append({'new':new,'base':base,'block_length':length,'year_equal_difference':float(v.mean()),'lower95':float(lo),'upper95':float(hi),'interval_width':float(hi-lo),'replicates':10000,'circular':False})
  draw_rows.extend({'new':new,'base':base,'block_length':length,'replicate':j,'year_equal_difference':float(x)} for j,x in enumerate(bs))
pd.DataFrame(intervals).to_csv(OUT/'block_length_sensitivity.csv',index=False);pd.DataFrame(draw_rows).to_csv(OUT/'block_bootstrap_draws.csv.gz',index=False,compression='gzip')
v=(annual.eco_F-annual.eco).to_numpy();v=v-v.mean();acf=[{'lag':lag,'sample_ACF':float(np.dot(v[:-lag],v[lag:])/np.dot(v,v)),'n_year_pairs':len(v)-lag} for lag in range(1,9)];pd.DataFrame(acf).to_csv(OUT/'paired_loss_ACF.csv',index=False)
lagdiag=[]
for plot,g in D.groupby('Plot'):
 lagdiag.append({'Plot':plot,'n_units':len(g),'log_DON_lag1_correlation':float(np.corrcoef(np.log1p(g.outcome),np.log1p(g.lag_DON))[0,1])})
pd.DataFrame(lagdiag).to_csv(OUT/'descriptive_lag_correlations.csv',index=False)

# Scottish site groups use fixed assay counts, selection quotas and tie rules.
s=pd.read_csv(FILES['scotland']);s.Site_code=s.Site_code.str.strip();scot=[];detail=[];conditional=[];zero=[];rng=np.random.default_rng(2026101432)
visitorder={'May':0,'July':1,'September':2}
for (yr,visit),g in sorted(s.groupby(['Year','Visit']),key=lambda t:(t[0][0],visitorder[t[0][1]])):
 g=g.sort_values('Site_code');quota=len(g)//3
 estimates={'raw':lambda gg:gg.k/gg.n,'Jeffreys_mean':lambda gg:(gg.k+.5)/(gg.n+1),'add_two':lambda gg:(gg.k+2)/(gg.n+4)}
 for scope in ['all_sites','positive_fixed_original_quota','positive_third_quota']:
  gg=g if scope=='all_sites' else g[g.k>0];m=quota if scope!='positive_third_quota' else len(gg)//3;assert len(gg)>=m>=1
  density=gg.DON_100.to_numpy();ids=gg.Site_code.to_numpy();chosen=np.lexsort((ids,-density))[:m]
  for estimate,fn in estimates.items():
   pp=np.asarray(fn(gg));risk=density*pp;best=np.lexsort((ids,-risk))[:m];overlap=len(set(chosen)&set(best));ret=float(risk[chosen].sum()/risk[best].sum())
   scot.append({'year':int(yr),'visit':visit,'scope':scope,'estimator':estimate,'n_sites':len(gg),'quota':m,'zero_positive_sites':int((gg.k==0).sum()),'overlap':overlap,'coverage':overlap/m,'retained_risk_ratio':ret,'DON_selected':','.join(ids[chosen]),'DIN_selected':','.join(ids[best])})
   detail.extend({'year':int(yr),'visit':visit,'scope':scope,'estimator':estimate,'Site_code':row.Site_code,'k':int(row.k),'n':int(row.n),'NIP_estimate':float(pp[j]),'DON_100':float(density[j]),'DIN_index':float(risk[j]),'selected_DON':j in chosen,'selected_DIN':j in best} for j,row in enumerate(gg.itertuples()))
 density=g.DON_100.to_numpy();ids=g.Site_code.to_numpy();chosen=np.lexsort((ids,-density))[:quota];pp=rng.beta(g.k.to_numpy()+.5,g.n.to_numpy()-g.k.to_numpy()+.5,(20000,len(g)));risks=pp*density;best=np.argsort(risks,axis=1)[:,-quota:];ratios=risks[:,chosen].sum(1)/np.take_along_axis(risks,best,axis=1).sum(1);same=(np.isin(best,chosen).sum(1)==quota)
 conditional.append({'year':int(yr),'visit':visit,'n_sites':len(g),'quota':quota,'conditional_mean_retained_risk':float(ratios.mean()),'conditional_lower95':float(np.quantile(ratios,.025)),'conditional_upper95':float(np.quantile(ratios,.975)),'probability_same_set':float(same.mean()),'replicates':20000})
 for r in g[g.k==0].itertuples():zero.append({'year':int(yr),'visit':visit,'Site_code':r.Site_code,'n':int(r.n),'posterior_mean':.5/(r.n+1),'posterior_lower95':float(beta_distribution.ppf(.025,.5,r.n+.5)),'posterior_upper95':float(beta_distribution.ppf(.975,.5,r.n+.5))})
pd.DataFrame(scot).to_csv(OUT/'Scottish_all_fixed_sensitivities.csv',index=False);pd.DataFrame(detail).to_csv(OUT/'Scottish_selection_details.csv',index=False);pd.DataFrame(conditional).to_csv(OUT/'Scottish_conditional_set_uncertainty.csv',index=False);pd.DataFrame(zero).to_csv(OUT/'Scottish_zero_positive_posterior_intervals.csv',index=False)
sc=pd.DataFrame(scot);sm=[]
for (scope,est),g in sc.groupby(['scope','estimator']):sm.append({'scope':scope,'estimator':est,'visits':len(g),'covered_slots':int(g.overlap.sum()),'total_slots':int(g.quota.sum()),'equal_visit_mean_coverage':float(g.coverage.mean()),'equal_visit_mean_retained_risk':float(g.retained_risk_ratio.mean()),'visits_different_sets':int((g.overlap<g.quota).sum())})
pd.DataFrame(sm).to_csv(OUT/'Scottish_sensitivity_summary.csv',index=False)

# Numerical stability of the exact observed-risk identity.
m=pd.read_csv(FILES['modern']);t=pd.read_csv(FILES['transitions']);t['minimum_endpoint_NIP']=t[['NIP0','NIP1']].min(axis=1);t.sort_values(['minimum_endpoint_NIP','to_year','Plot']).head(10).to_csv(OUT/'ten_lowest_prevalence_transitions.csv',index=False)
residual=t.delta_DIN-t.quantity_contribution-t.infection_contribution
riskcheck={'n_modern_units':len(m),'n_transitions':len(t),'minimum_NIP':float(m.NIP.min()),'minimum_DON':float(m.DON_mean.min()),'zero_NIP_units':int((m.NIP==0).sum()),'zero_DON_units':int((m.DON_mean==0).sum()),'maximum_identity_residual':float(abs(residual).max()),'difference_formula_divides_by_P_or_D':False}

# Fixed PCR windows at unchanged annual density: a risk index, not seasonal measured DIN.
pcr=pd.read_csv(FILES['pcr']);pcr['Plot']=pcr.Grid.map({'greenC':'GC','greenE':'GE','henryC':'HC','henryE':'HE','teaC':'TC','teaE':'TE'});windows=[];ranks=[]
for label,months in [('May_July',[5,6,7]),('June_August',[6,7,8])]:
 z=pcr[pcr.month.isin(months)].groupby(['Year','Plot'])['B.burgdorferi'].agg(['sum','count']).reset_index().rename(columns={'Year':'outcome_year','sum':'window_k','count':'window_n'});z=m[['outcome_year','Plot','DON_mean']].merge(z,on=['outcome_year','Plot'],validate='one_to_one');z['window']=label;z['window_NIP']=z.window_k/z.window_n;z['window_DIN_index']=z.DON_mean*z.window_NIP;windows.append(z)
 for yr,g in z.groupby('outcome_year'):
  g=g.sort_values('Plot');ids=g.Plot.to_numpy();dn=np.lexsort((ids,-g.DON_mean.to_numpy()))[:2];di=np.lexsort((ids,-g.window_DIN_index.to_numpy()))[:2];ranks.append({'year':int(yr),'window':label,'available_grids':len(g),'k':int(g.window_k.sum()),'n':int(g.window_n.sum()),'zero_positive_grids':int((g.window_k==0).sum()),'overlap':len(set(dn)&set(di)),'quota':2,'DON_selected':','.join(ids[dn]),'DIN_selected':','.join(ids[di])})
w=pd.concat(windows);rr=pd.DataFrame(ranks);common=sorted(set(rr[(rr.window=='May_July')&(rr.available_grids==6)].year)&set(rr[(rr.window=='June_August')&(rr.available_grids==6)].year));rr['common_complete_year']=rr.year.isin(common);w.to_csv(OUT/'window_specific_infection_indices.csv',index=False);rr.to_csv(OUT/'window_priority_sensitivity.csv',index=False)
ws=[]
for name,g in rr[rr.common_complete_year].groupby('window'):ws.append({'window':name,'common_years':common,'n_years':len(g),'covered_slots':int(g.overlap.sum()),'total_slots':int(g.quota.sum()),'different_set_years':int((g.overlap<g.quota).sum()),'total_PCR':int(g.n.sum())})
# Check that input files are unchanged.
assert all(sha(FILES[k])==v['sha256'] for k,v in hashes.items());assert bool((pred.train_last_year<pred.year).all());assert pred.groupby('model').size().eq(136).all()
summary={'primary_new_contrast':'eco_AR_F minus eco_AR','models':met,'contrasts':contrasts,'block_intervals':intervals,'original_prediction_max_absolute_difference':md,'new_ridge_fits_including_inner_folds':fit_count,'Scottish_sensitivity_summary':sm,'Scottish_conditional':conditional,'risk_identity_audit':riskcheck,'PCR_window_summary':ws,'analysis_scope':'Fixed model comparisons and sensitivity analyses.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'RUN_MANIFEST.json').write_text(json.dumps({'inputs':hashes,'script_sha256':sha(Path(__file__)),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'strict_past_only_training':True,'all_original_inputs_unchanged':True},indent=2));print(json.dumps(summary,indent=2),flush=True)
