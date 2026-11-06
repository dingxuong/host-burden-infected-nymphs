# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv,math,time
import numpy as np
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;O=P/'results';profiles=json.load(open(P/'inputs/profiles.json'));allrows=list(csv.DictReader(open(P/'inputs/previous_intervals.csv')));rows=[r for r in allrows if r['conditional_upper_below_099']=='True'];assert len(rows)==17
records=[];results=[];rep=[];saved={};start=time.time()
def solve(obj,A,b,eq,tau_cap_index,tag):
 res=linprog(-obj,A_ub=A,b_ub=b,A_eq=eq[None,:],b_eq=[1.],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if res.status==2:records.append(dict(tag=tag,status='infeasible'));return None
 assert res.success,(tag,res.message)
 violation=max(0.,float(np.max(A@res.x-b)),float(abs(eq@res.x-1)),float(-res.x.min()));assert violation<1e-7
 y=float(res.eqlin.marginals[0]);z=np.minimum(res.ineqlin.marginals,0.);c=-obj
 slack=c-y*eq-A.T@z
 correction=max(0.,float(np.max(-slack[:-1])))+1e-11;z[0]-=correction
 slack=c-y*eq-A.T@z;z[tau_cap_index]-=max(0.,float(-slack[-1]))+1e-11
 slack=c-y*eq-A.T@z;bound=-(y+b@z);witness=float(obj@res.x);gap=float(bound-witness)
 assert slack.min()>-1e-9 and -1e-8<gap<1e-5
 records.append(dict(tag=tag,status='optimal',constraint_violation=violation,dual_gap=gap,min_dual_slack=float(slack.min())))
 return dict(upper=float(bound),witness=witness,x=res.x,y=y,dual=z)
def coeff(z,w,G,knots):
 wh=w*G;pre=np.r_[0.,np.cumsum(wh*z)];suf=np.r_[np.cumsum(wh[::-1])[::-1],0.];ii=np.searchsorted(z,knots,side='right');return np.r_[math.fsum(wh),pre[ii]/knots+suf[ii],0.]
def make_constraints(a,c,pL,pU,qL,qU,l,u):
 n=len(a);zeros=np.zeros(n);eq=np.r_[1.,a,0.]
 A=np.row_stack([np.r_[1.,np.ones(n),-1.],np.r_[1.,zeros,-qU],np.r_[-1.,zeros,qL],np.r_[1.,c,-u],np.r_[-1.,-c,l],np.r_[0.,zeros,1.],np.r_[0.,zeros,-1.]])
 b=np.array([0.,0.,0.,0.,0.,1/pL,-1/pU]);return A,b,eq
for j,row in enumerate(rows):
 sp=row['species'];fam=row['feeding_family'];f0=float(row['f']);p0=float(row['p']);q00=float(row['floor_fraction'])*p0;loc=float(row['location']);l=float(row['measurement_low']);u=float(row['measurement_upper']);d=profiles['animal_augsep/'+sp];z=np.array(d['z']);w=np.array(d['w']);w=w/math.fsum(w);mu=math.fsum(z*w);v=z/mu;knots=np.unique(np.r_[z[z>0],mu,loc*mu]);a=np.minimum(mu/knots,1.);c=np.minimum(loc*mu/knots,1.)
 def G(f):
  if fam=='reciprocal':return v/(f+(1-f)*v)
  if fam=='exponential':return v*f**(v-1)
  return v/np.sqrt(v+(1-v)*f*f)
 A,b,eq=make_constraints(a,c,p0,p0,q00,q00,l,u);ob=coeff(z,w,G(f0),knots);rr=solve(ob,A,b,eq,5,'rep_%d'%j);assert rr is not None;diff=abs(rr['upper']-float(row['conditional_RI_upper']));assert diff<1e-7
 rep.append(dict(row=j,case=row['case'],difference=diff,original_upper=float(row['conditional_RI_upper']),replicated_upper=rr['upper']))
 for delta in [.01,.025,.05]:
  fL=max(.000001,f0-delta);fU=min(.999999,f0+delta);qL=max(0.,q00-delta);qU=min(1.,q00+delta);grid=np.linspace(fL,fU,9)
  for strategy in ['H','HM']:
   pL=p0-delta;pU=p0+delta
   if strategy=='HM':pL=max(pL,p0-.01);pU=min(pU,p0+.01)
   A,b,eq=make_constraints(a,c,pL,pU,qL,qU,l,u);tag='row%d_d%s_%s'%(j,delta,strategy);end=[];env=[]
   for fi,f in enumerate(grid):
    obj=coeff(z,w,G(float(f)),knots);rr=solve(obj,A,b,eq,5,tag+'_f'+str(fi));assert rr is not None;end.append((rr,float(f)))
   for si,(fl,fu) in enumerate(zip(grid[:-1],grid[1:])):
    obj=coeff(z,w,np.maximum(G(float(fl)),G(float(fu))),knots);rr=solve(obj,A,b,eq,5,tag+'_s'+str(si));assert rr is not None;env.append((rr,si,obj))
   worst=max(env,key=lambda x:x[0]['upper']);best=max(end,key=lambda x:x[0]['witness']);upper=worst[0]['upper'];witness=best[0]['witness'];assert upper>=witness-1e-7
   status='down' if upper<.99-1e-7 else ('not_guaranteed' if witness>=.99+1e-7 else 'envelope_unresolved')
   info={k:row[k] for k in ['case','species','feeding_family','f','p','floor_fraction','location','reference_curve','halfwidth','hypothetical_center']};info.update(row=j,delta=delta,strategy=strategy,original_upper=float(row['conditional_RI_upper']),f_low=fL,f_high=fU,p_low=pL,p_high=pU,q0_low=qL,q0_high=qU,measurement_low=l,measurement_upper=u,upper=upper,endpoint_witness=witness,cover_gap=upper-witness,status=status,worst_subinterval=worst[1],worst_endpoint_f=best[1]);results.append(info)
   for suf,res,obj in [('envelope',worst[0],worst[2]),('endpoint',best[0],coeff(z,w,G(best[1]),knots))]:
    for name,val in dict(x=res['x'],dual=res['dual'],y=np.array([res['y']]),obj=obj).items():saved[tag+'_'+suf+'_'+name]=val
   saved[tag+'_knots']=knots;saved[tag+'_A']=A;saved[tag+'_b']=b;saved[tag+'_eq']=eq
 print('completed',j+1,'/17',flush=True)
def write(name,rr):
 with open(O/name,'w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
write('robustness.csv',results);write('replication.csv',rep)
with open(O/'solver_records.csv','w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=['tag','status','constraint_violation','dual_gap','min_dual_slack']);wr.writeheader();wr.writerows(records)
np.savez_compressed(O/'witnesses.npz',**saved)
summary=dict(scope='Output bounds with uncertain response probabilities at the measurement anchors',original_positive_combinations=17,range_strategy_combinations=len(results),solver_calls=len(records),elapsed_seconds=time.time()-start)
summary['counts']={str(d):{s:{st:sum(r['delta']==d and r['strategy']==s and r['status']==st for r in results) for st in ['down','not_guaranteed','envelope_unresolved']} for s in ['H','HM']} for d in [.01,.025,.05]}
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
