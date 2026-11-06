# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;O=P/'results';D=list(csv.DictReader(open(O/'robustness.csv')));W=np.load(O/'witnesses.npz');profiles=json.load(open(P/'inputs/profiles.json'));dt=np.longdouble;maxobj=maxbound=maxconstraint=0.;minslack=1.
for r in D:
 tag='row%d_d%s_%s'%(int(r['row']),float(r['delta']),r['strategy']);sp=r['species'];f0=dt(r['f']);loc=dt(r['location']);z=np.array(profiles['animal_augsep/'+sp]['z'],dtype=dt);w=np.array(profiles['animal_augsep/'+sp]['w'],dtype=dt);w=w/w.sum();mu=(w*z).sum();u=z/mu;k=np.array(W[tag+'_knots'],dtype=dt);a=np.minimum(mu/k,1);c=np.minimum(loc*mu/k,1);n=len(k);zero=np.zeros(n,dtype=dt)
 pL=dt(r['p_low']);pU=dt(r['p_high']);qL=dt(r['q0_low']);qU=dt(r['q0_high']);l=dt(r['measurement_low']);hi=dt(r['measurement_upper']);A=np.row_stack([np.r_[dt(1),np.ones(n,dtype=dt),dt(-1)],np.r_[dt(1),zero,-qU],np.r_[dt(-1),zero,qL],np.r_[dt(1),c,-hi],np.r_[dt(-1),-c,l],np.r_[dt(0),zero,dt(1)],np.r_[dt(0),zero,dt(-1)]]);b=np.array([0,0,0,0,0,1/pL,-1/pU],dtype=dt);eq=np.r_[dt(1),a,dt(0)]
 assert np.max(abs(A-W[tag+'_A']))<dt('1e-8')
 def G(f):
  if r['feeding_family']=='reciprocal':return u/(f+(1-f)*u)
  if r['feeding_family']=='exponential':return u*f**(u-1)
  return u/np.sqrt(u+(1-u)*f*f)
 for mode in ['endpoint','envelope']:
  x=np.array(W[tag+'_'+mode+'_x'],dtype=dt);du=np.array(W[tag+'_'+mode+'_dual'],dtype=dt);y=dt(W[tag+'_'+mode+'_y'][0]);tau=x[-1];p=1/tau;q0=x[0]/tau;alpha=x[1:-1]/tau
  assert pL-dt('1e-7')<=p<=pU+dt('1e-7') and qL-dt('1e-7')<=q0<=qU+dt('1e-7')
  maxconstraint=max(maxconstraint,float(max(np.max(A@x-b),abs(eq@x-1),-x.min(),0)))
  # Compute output by summing actual support responses, independently of stored coefficients.
  qt=np.full(len(z),x[0],dtype=dt)
  for j in np.where(x[1:-1]>dt('1e-12'))[0]:qt+=x[j+1]*np.minimum(z/k[j],1)
  if mode=='endpoint':g=G(dt(r['worst_endpoint_f']))
  else:
   grid=np.linspace(dt(r['f_low']),dt(r['f_high']),9,dtype=dt);idx=int(r['worst_subinterval']);g=np.maximum(G(grid[idx]),G(grid[idx+1]))
  actual=(w*g*qt).sum();obj=np.array(W[tag+'_'+mode+'_obj'],dtype=dt);maxobj=max(maxobj,float(abs(actual-obj@x)))
  slack=-obj-y*eq-A.T@du;minslack=min(minslack,float(slack.min()));assert np.max(du)<dt('1e-9')
  if mode=='envelope':maxbound=max(maxbound,float(abs(-(y+b@du)-dt(r['upper']))))
 assert float(r['upper'])>=float(r['endpoint_witness'])-1e-7
 if r['status']=='down':assert float(r['upper'])<.99-1e-7
 elif r['status']=='not_guaranteed':assert float(r['endpoint_witness'])>=.99+1e-7
assert maxobj<1e-8 and maxbound<1e-8 and maxconstraint<1e-7 and minslack>-1e-8

print("Independent finite-LP arithmetic checks passed")
