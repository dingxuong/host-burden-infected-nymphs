# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv,math,time
import numpy as np
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;O=P/'results';start=time.time();profiles=json.load(open(P/'inputs/profiles.json'));original=list(csv.DictReader(open(P/'inputs/original_C1.csv',encoding='utf-8-sig')));locations=[.25,.5,2.,4.];cutoff=.99;tol=1e-9;logs=[];arrays={};base_rows=[];design=[];conditioned=[]
def solve(c,A,b,a,target,tag):
 r=linprog(c,A_ub=A,b_ub=b,A_eq=a[None,:],b_eq=[target],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':tol,'dual_feasibility_tolerance':tol})
 if r.status==2:logs.append(dict(tag=tag,status='infeasible'));return None
 assert r.success,(tag,r.message)
 v=max(0.,float(np.max(A@r.x-b)),float(abs(a@r.x-target)),float(-r.x.min()));assert v<1e-7
 y=float(r.eqlin.marginals[0]);dual=np.minimum(r.ineqlin.marginals,0.);repair=max(0.,float(np.max(a*y+A.T@dual-c)))+1e-11;dual[0]-=repair
 slack=c-a*y-A.T@dual;lo=target*y+b@dual;gap=float(r.fun-lo);assert slack.min()>-1e-10 and gap>-1e-8 and gap<2e-6
 logs.append(dict(tag=tag,status='optimal',constraint_violation=v,dual_gap=gap,min_dual_slack=float(slack.min())))
 return dict(x=r.x,primal=float(r.fun),dual=float(lo),y=y,multipliers=dual)
def coeff(z,w,h,knots):
 wh=w*h;prefix=np.r_[0,np.cumsum(wh*z)];suffix=np.r_[np.cumsum(wh[::-1])[::-1],0.];ix=np.searchsorted(z,knots,side='right');return prefix[ix]/knots+suffix[ix]
for ci,row in enumerate(original):
 sp=row['species'];fam=row['feeding_family'];f=float(row['f']);p=float(row['p']);eta=float(row['floor_fraction']);q0=eta*p;key='case%02d'%ci
 data=profiles['animal_augsep/'+sp];z=np.array(data['z']);w=np.array(data['w']);w=w/math.fsum(w);mu=math.fsum(z*w);u=z/mu
 if fam=='reciprocal':s=1/(1+(1/f-1)*u)
 elif fam=='exponential':s=np.exp(np.log(f)*u)
 else:s=1/np.sqrt(1+(f**-2-1)*u)
 h=z*s;eh=math.fsum(w*h);den=mu*f*p;offset=q0*eh/den;target=p-q0;cap=1-q0
 knots=np.unique(np.r_[z[z>0],mu]);a=np.minimum(mu/knots,1.);B=coeff(z,w,h,knots)/den
 r=solve(-B,np.ones((1,len(knots))),np.array([cap]),a,target,key+'_base');upper=offset-r['primal'];lower=offset+target*math.fsum(w*h*np.minimum(z/mu,1))/den
 assert abs(upper-float(row['upper_R_I']))<1e-7 and abs(lower-float(row['lower_R_I']))<1e-7
 base_rows.append(dict(case=key,species=sp,feeding_family=fam,f=f,p=p,floor_fraction=eta,lower=lower,upper=upper,direction=row['direction']))
 if row['direction']=='guaranteed_down':continue
 for loc in locations:
  t=loc*mu;knots=np.unique(np.r_[z[z>0],mu,t]);a=np.minimum(mu/knots,1.);B=coeff(z,w,h,knots)/den;c=np.minimum(t/knots,1.);A=np.ones((1,len(knots)));b=np.array([cap]);tag=key+'_t'+str(loc)
  rmin=solve(c,A,b,a,target,tag+'_qmin');rmax=solve(-c,A,b,a,target,tag+'_qmax');qmin=q0+rmin['primal'];qmax=q0-rmax['primal']
  expectedmin=q0+target*min(loc,1.);expectedmax=min(1.,q0+target*loc) if loc>1 else p
  assert abs(qmin-expectedmin)<1e-7 and abs(qmax-expectedmax)<1e-7
  dangerA=np.row_stack([A,-B]);dangerb=np.r_[cap,-(cutoff-offset)];rd=solve(c,dangerA,dangerb,a,target,tag+'_danger');assert rd is not None
  threshold=q0+rd['dual'];witness=q0+rd['primal'];window=max(threshold-qmin,0.);risk=offset+B@rd['x'];assert risk>=cutoff-1e-7
  design.append(dict(case=key,species=sp,feeding_family=fam,f=f,p=p,floor_fraction=eta,location=loc,q_min=qmin,q_max=qmax,danger_q_lower=threshold,danger_q_witness=witness,informative_window=window,danger_RI_witness=float(risk)))
  arrays[tag+'_knots']=knots;arrays[tag+'_danger_alpha']=rd['x'];arrays[tag+'_B']=B;arrays[tag+'_a']=a;arrays[tag+'_c']=c;arrays[tag+'_dual']=rd['multipliers'];arrays[tag+'_dual_y']=np.array([rd['y']])
  for curve in ['rational','exponential']:
   br=(p-q0)/(1-p)
   qr=q0+(1-q0)*br*u/(1+br*u) if curve=='rational' else q0+(1-q0)*(1-np.exp(np.log((1-p)/(1-q0))*u))
   qt=q0+(1-q0)*br*loc/(1+br*loc) if curve=='rational' else q0+(1-q0)*(1-math.exp(math.log((1-p)/(1-q0))*loc))
   originalRI=math.fsum(w*h*qr)/den
   for eps in [.01,.025,.05]:
    l=max(q0,qt-eps);v=min(1.,qt+eps);C=np.row_stack([A,c,-c]);D=np.r_[cap,v-q0,-(l-q0)];up=solve(-B,C,D,a,target,tag+'_'+curve+'_'+str(eps));assert up is not None
    upper2=offset-up['dual'];primalupper=offset-up['primal'];assert primalupper<=upper+1e-7 and originalRI<=upper2+1e-7
    criterion=v<threshold-1e-7
    if criterion:assert upper2<cutoff
    conditioned.append(dict(case=key,species=sp,feeding_family=fam,f=f,p=p,floor_fraction=eta,location=loc,reference_curve=curve,hypothetical_center=qt,halfwidth=eps,measurement_low=l,measurement_upper=v,threshold=threshold,center_to_threshold=threshold-qt,reference_RI=originalRI,conditional_RI_upper=upper2,conditional_RI_upper_witness=primalupper,threshold_certifies_down=criterion,conditional_upper_below_099=upper2<cutoff-1e-7))
 print(key,'complete',row['direction'],flush=True)
def write(name,rows):
 with open(O/name,'w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
write('baseline_reproduction.csv',base_rows);write('measurement_thresholds.csv',design);write('hypothetical_intervals.csv',conditioned);write('solver_records.csv',[{k:r.get(k,'') for k in ['tag','status','constraint_violation','dual_gap','min_dual_slack']} for r in logs]);np.savez_compressed(O/'danger_witnesses.npz',**arrays)
summary=dict(scope='Conditional C1 output bounds under hypothetical measurements',baseline_cases=len(base_rows),original_undetermined=sum(r['direction']=='not_identified' for r in base_rows),locations_per_case=4,threshold_rows=len(design),informative_locations=sum(r['informative_window']>1e-7 for r in design),interval_scenarios=len(conditioned),intervals_resolved_down=sum(r['conditional_upper_below_099'] for r in conditioned),elapsed_seconds=time.time()-start)
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
