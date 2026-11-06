from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);args=ap.parse_args()
BASE=args.inputs;O=args.results;I=json.loads((O/'RUN_MANIFEST.json').read_text())['inputs']
for key in I:I[key]['path']=str(BASE/(key+'.csv'))
ss=pd.read_csv(O/'all_126_site_memberships.csv');gg=pd.read_csv(O/'all_15_group_results.csv');aa=pd.read_csv(O/'system_equal_group_results.csv');summary=np.load(O/'all_summary_draws.npz');errors=[];checks=0
for r in gg.itertuples():
 s=ss[(ss.system==r.system)&(ss.year==r.year)&(ss.visit==r.visit)].sort_values('site');key=f'{r.system}_{r.year}_{r.visit}';z=np.load(O/(key+'_draws.npz'));P=z['P'];replicate_k=z['replicate_k'];null_k=z['null_k'];D=s.D_per450m2.to_numpy();k=s.k.to_numpy();n=s.n.to_numpy();ids=s.site.to_numpy();m=r.quota
 # Recalculate ranks with Python sorting.
 get=lambda risk:sorted(range(len(ids)),key=lambda j:(-risk[j],ids[j]))[:m]
 d=get(D);j=get(D*(k+.5)/(n+1));raw=get(D*k/n);assert sum(x in d for x in raw)==r.raw_coverage
 gain=np.zeros(len(P));oracle=np.zeros(len(P));inc=np.zeros(len(ids));repinc=np.zeros(len(ids));rep_same=0;nullcover=[];rawnull=[]
 for b in range(len(P)):
  risk=D*P[b];q=get(risk);oracle[b]=sum(risk[x] for x in q);gain[b]=sum(risk[x] for x in j)-sum(risk[x] for x in d);inc[q]+=1
  jk=get(D*(replicate_k[b]+.5)/(n+1));repinc[jk]+=1;rep_same+=set(jk)==set(j)
  nj=get(D*(null_k[b]+.5)/(n+1));nr=get(D*null_k[b]/n);nullcover.append(sum(x in d for x in nj));rawnull.append(sum(x in d for x in nr))
 np.testing.assert_allclose(gain,summary[key+'_delta'],atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(inc/len(P),s.latent_oracle_inclusion_probability,atol=1e-12)
 np.testing.assert_allclose(repinc/len(P),s.repeat_PCR_Jeffreys_inclusion_probability,atol=1e-12)
 np.testing.assert_array_equal(nullcover,summary[key+'_null_Jeffreys_coverage']);np.testing.assert_array_equal(rawnull,summary[key+'_null_raw_coverage'])
 assert abs(rep_same/len(P)-r.replicate_PCR_same_Jeffreys_set)<1e-12
 for attr,quant in [('delta_lower95',.025),('delta_upper95',.975)]:assert abs(np.quantile(gain,quant)-getattr(r,attr))<1e-10
 assert all((np.isfinite(z[nm]).all() for nm in z.files));checks+=len(P)
for r in aa.itertuples():
 sub=gg[gg.system==r.system];v=np.mean([summary[f'{r.system}_{x.year}_{x.visit}_delta'] for x in sub.itertuples()],axis=0);assert abs(np.quantile(v,.025)-r.delta_lower95)<1e-10;assert abs(v.mean()-r.delta_mean)<1e-10
manifest=json.loads((O/'RUN_MANIFEST.json').read_text());assert all(hashlib.sha256((O/k).read_bytes()).hexdigest()==v for k,v in manifest['outputs'].items());assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in I.values())
report={'passed':True,'independent_python_sorted_draws_checked':checks,'groups':len(gg),'site_records':len(ss),'posterior_repeat_and_null_arrays_verified':True,'input_and_output_hashes_match':True,'new_randomness_or_model_fits':0};(O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(report)
