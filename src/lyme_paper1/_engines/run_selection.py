'Evaluate infection-informed site selection using repeated PCR counts at fixed DON.'
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,platform,datetime
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);args=ap.parse_args()
BASE=args.inputs;O=args.output;O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
I={key:{'path':str(BASE/(key+'.csv')),'sha256':sha(BASE/(key+'.csv'))} for key in ['cary','scotland']}
c=pd.read_csv(I['cary']['path']);s=pd.read_csv(I['scotland']['path']);groups=[]
for y,g in c.groupby('outcome_year',sort=True):groups.append(('Cary',int(y),'annual',g[['Plot','k','n','DON_mean']].rename(columns={'Plot':'site','DON_mean':'D'}),2))
for y in sorted(s.Year.unique()):
 for visit in ['May','July','September']:
  g=s[(s.Year==y)&(s.Visit==visit)][['Site_code','k','n','DON_100']].rename(columns={'Site_code':'site','DON_100':'D'}).copy();g['D']*=4.5;groups.append(('Scotland',int(y),visit,g,len(g)//3))
rp=np.random.default_rng(2026101701);rr=np.random.default_rng(2026101702);rn=np.random.default_rng(2026101703);B=20000;rows=[];sites=[];sets=[];draws={};sysdraw={};nulls={};checks=[]
# Stable sort: input is alphabetical and tied scores preserve that order.
top=lambda x,m:np.argsort(-np.asarray(x),axis=-1,kind='stable')[...,:m]
summary=lambda x:{'mean':float(np.mean(x)),'lower95':float(np.quantile(x,.025)),'upper95':float(np.quantile(x,.975))}
for system,year,visit,g,m in groups:
 g=g.sort_values('site').reset_index(drop=True);ids=g.site.to_numpy();D=g.D.to_numpy();k=g.k.to_numpy();n=g.n.to_numpy();assert len(g)>m>0 and np.all(D>0) and np.all(n>0) and np.all((k>=0)&(k<=n)) and np.all(k==k.astype(int)) and np.all(n==n.astype(int))
 k=k.astype(int);n=n.astype(int);pmean=(k+.5)/(n+1);sd=top(D,m);sj=top(D*pmean,m);raw=top(D*k/n,m)
 P=rp.beta(k+.5,n-k+.5,(B,len(g)));risk=P*D;oracle=top(risk,m);orisk=np.take_along_axis(risk,oracle,axis=1).sum(1)
 ud=risk[:,sd].sum(1);uj=risk[:,sj].sum(1);delta=uj-ud;weight=np.isin(np.arange(len(g)),sj).astype(int)-np.isin(np.arange(len(g)),sd).astype(int);alt=(risk*weight).sum(1);err=float(abs(delta-alt).max());assert err<1e-10
 mean_exact=float((D*pmean*weight).sum());var_exact=float(np.sum((D*weight)**2*((k+.5)*(n-k+.5)/((n+1)**2*(n+2)))));mc_se=np.sqrt(var_exact/B);assert abs(delta.mean()-mean_exact)<8*mc_se+1e-12
 membership=np.zeros_like(P,dtype=bool);np.put_along_axis(membership,oracle,True,axis=1)
 replicated=rr.binomial(n,P);rep_sel=top(D*(replicated+.5)/(n+1),m);rep_membership=np.zeros_like(membership);np.put_along_axis(rep_membership,rep_sel,True,axis=1)
 sj_mask=np.isin(np.arange(len(g)),sj);sd_mask=np.isin(np.arange(len(g)),sd)
 p0=float(k.sum()/n.sum());knull=rn.binomial(n,p0,(B,len(g)));nj=top(D*(knull+.5)/(n+1),m);nr=top(D*knull/n,m);coverj=np.isin(nj,sd).sum(1);coverr=np.isin(nr,sd).sum(1)
 key=f'{system}_{year}_{visit}';obsj=int(np.isin(sj,sd).sum());obsraw=int(np.isin(raw,sd).sum());row={'system':system,'year':year,'visit':visit,'n_sites':len(g),'quota':m,'raw_coverage':obsraw,'Jeffreys_coverage':obsj,'analytic_delta_mean':mean_exact,'delta_MC_standard_error':mc_se,'probability_gain':float(np.mean(delta>1e-12)),'probability_loss':float(np.mean(delta< -1e-12)),'probability_equal':float(np.mean(abs(delta)<=1e-12)),'DON_probability_oracle_set':float(np.mean(np.all(membership==sd_mask,axis=1))),'Jeffreys_probability_oracle_set':float(np.mean(np.all(membership==sj_mask,axis=1))),'replicate_PCR_same_Jeffreys_set':float(np.mean(np.all(rep_membership==sj_mask,axis=1))),'null_common_prevalence':p0,'null_Jeffreys_mean_coverage':float(coverj.mean()),'null_Jeffreys_same_set':float(np.mean(coverj==m))}
 for label,x in [('delta',delta),('DON_retention',ud/orisk),('Jeffreys_retention',uj/orisk)]:row.update({label+'_'+name:v for name,v in summary(x).items()})
 rows.append(row);sets.append({'system':system,'year':year,'visit':visit,'DON_selected':','.join(ids[sd]),'Jeffreys_selected':','.join(ids[sj]),'raw_DIN_selected':','.join(ids[raw])})
 for j,id in enumerate(ids):sites.append({'system':system,'year':year,'visit':visit,'site':id,'D_per450m2':D[j],'k':k[j],'n':n[j],'posterior_mean_prevalence':pmean[j],'DON_selected':bool(sd_mask[j]),'fixed_Jeffreys_selected':bool(sj_mask[j]),'latent_oracle_inclusion_probability':float(membership[:,j].mean()),'repeat_PCR_Jeffreys_inclusion_probability':float(rep_membership[:,j].mean())})
 draws[key+'_delta']=delta;draws[key+'_DON_retention']=ud/orisk;draws[key+'_Jeffreys_retention']=uj/orisk;draws[key+'_null_Jeffreys_coverage']=coverj;draws[key+'_null_raw_coverage']=coverr
 sysdraw.setdefault(system,[]).append({'delta':delta,'analytic_delta':mean_exact,'DON_retention':ud/orisk,'Jeffreys_retention':uj/orisk});nulls.setdefault(system,[]).append((coverj,coverr,obsj,obsraw,m));checks.append({'key':key,'delta_matrix_max_error':err,'analytic_mean_MC_z':float((delta.mean()-mean_exact)/mc_se) if mc_se else 0.,'selected_fixed_before_draws':True})
 # Save infection proportions and simulated counts for independent checking.
 np.savez_compressed(O/(key+'_draws.npz'),P=P,replicate_k=replicated,null_k=knull)
aggregate=[]
for system,values in sysdraw.items():
 row={'system':system,'groups':len(values),'analytic_delta_mean':float(np.mean([v['analytic_delta'] for v in values]))}
 for label in ['delta','DON_retention','Jeffreys_retention']:
  v=np.mean([g[label] for g in values],axis=0);draws[system+'_aggregate_'+label]=v;row.update({label+'_'+name:x for name,x in summary(v).items()})
  if label=='delta':row.update(probability_gain=float(np.mean(v>1e-12)),probability_loss=float(np.mean(v< -1e-12)),probability_equal=float(np.mean(abs(v)<=1e-12)))
 nullj=np.sum([x[0] for x in nulls[system]],axis=0);nullr=np.sum([x[1] for x in nulls[system]],axis=0);total=sum(x[4] for x in nulls[system]);obsj=sum(x[2] for x in nulls[system]);obsr=sum(x[3] for x in nulls[system]);row.update(total_slots=total,observed_Jeffreys_coverage=obsj,observed_raw_coverage=obsr)
 for label,v,obs in [('null_Jeffreys',nullj,obsj),('null_raw',nullr,obsr)]:
  row.update({label+'_'+name:x for name,x in summary(v).items()});row[label+'_fraction_at_most_observed']=float(np.mean(v<=obs));draws[system+'_'+label+'_total']=v
 assert obsr==({'Cary':14,'Scotland':10}[system]);aggregate.append(row)
pd.DataFrame(rows).to_csv(O/'all_15_group_results.csv',index=False);pd.DataFrame(sites).to_csv(O/'all_126_site_memberships.csv',index=False);pd.DataFrame(sets).to_csv(O/'fixed_selection_sets.csv',index=False);pd.DataFrame(aggregate).to_csv(O/'system_equal_group_results.csv',index=False);np.savez_compressed(O/'all_summary_draws.npz',**draws)
manifest={'inputs':I,'UTC_time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'replicates':B,'seeds':[2026101701,2026101702,2026101703],'scope':'Expected selection gain conditional on observed DON and the Jeffreys infection model.','group_count':len(rows),'site_count':len(sites),'checks':checks,'outputs':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(pd.DataFrame(aggregate).to_string(index=False))
