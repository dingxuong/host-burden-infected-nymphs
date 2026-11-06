# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;O=P/'results';profiles=json.load(open(P/'inputs/profiles.json'));D=list(csv.DictReader(open(O/'measurement_thresholds.csv')));W=np.load(O/'danger_witnesses.npz');maxdual=0.;maxobj=0.;maxcoeff=0.;minslack=1.;dt=np.longdouble
for r in D:
 tag=r['case']+'_t'+str(float(r['location']));z=np.array(profiles['animal_augsep/'+r['species']]['z'],dtype=dt);w=np.array(profiles['animal_augsep/'+r['species']]['w'],dtype=dt);w=w/w.sum();mu=(w*z).sum();u=z/mu;f=dt(r['f']);p=dt(r['p']);q0=dt(r['floor_fraction'])*p
 if r['feeding_family']=='reciprocal':s=1/(1+(1/f-1)*u)
 elif r['feeding_family']=='exponential':s=np.exp(np.log(f)*u)
 else:s=1/np.sqrt(1+(f**-2-1)*u)
 h=z*s;hw=h*w;den=mu*f*p;off=q0*hw.sum()/den;k=np.array(W[tag+'_knots'],dtype=dt);a=np.minimum(mu/k,1);c=np.minimum(mu*dt(r['location'])/k,1)
 # Independent reverse-tail formula, longdouble rather than stored double prefix sums.
 tail=np.r_[np.cumsum(hw[::-1])[::-1],dt(0)];tailz=np.r_[np.cumsum((hw*z)[::-1])[::-1],dt(0)];ix=np.searchsorted(z,k,side='right');B=((tailz[0]-tailz[ix])/k+tail[ix])/den
 maxcoeff=max(maxcoeff,float(np.max(abs(B-W[tag+'_B']))));alpha=np.array(W[tag+'_danger_alpha'],dtype=dt);y=dt(W[tag+'_dual_y'][0]);du=np.array(W[tag+'_dual'],dtype=dt)
 # Dual for min c alpha subject to sum alpha<=cap, -B alpha<=off-.99, a alpha=target.
 slack=c-y*a-du[0]+du[1]*B;minslack=min(minslack,float(slack.min()));bound=q0+(p-q0)*y+(1-q0)*du[0]+(off-dt('.99'))*du[1]
 maxdual=max(maxdual,abs(float(bound)-float(r['danger_q_lower'])))
 active=np.where(alpha>dt('1e-12'))[0];q=np.full(len(z),q0,dtype=dt)
 for j in active:q+=alpha[j]*np.minimum(z/k[j],1)
 RI=(hw*q).sum()/den;maxobj=max(maxobj,abs(float(RI)-float(r['danger_RI_witness'])))
 assert RI>=dt('.99')-dt('1e-7') and abs(a@alpha-(p-q0))<dt('1e-7') and alpha.sum()<=1-q0+dt('1e-7')
 assert np.min(alpha)>=-dt('1e-9') and abs(float(q0+c@alpha)-float(r['danger_q_witness']))<1e-7
assert maxcoeff<1e-8 and maxdual<1e-8 and maxobj<1e-8 and minslack>-1e-8

print("Independent finite-LP arithmetic checks passed")
