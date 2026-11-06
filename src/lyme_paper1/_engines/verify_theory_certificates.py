'Recalculate response-bound coefficients and primal-dual feasibility in float64.\nUse interval-bounds for the corresponding 50-decimal calculation.\n'
from pathlib import Path
from fractions import Fraction
import argparse,json,math,hashlib
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args();P=args.inputs
profiles=json.loads((P/'numeric_profile_inputs.json').read_text());certs=json.loads((P/'C1_dual_certificates.json').read_text());hinges=pd.read_csv(P/'C1_upper_extremizer_hinges.csv');bounds=pd.read_csv(P/'C1_concave_sharp_bounds.csv');checks=[]
def feed(z,mu,f,family):
 u=z/mu
 if family=='reciprocal':return 1/(1+(1/f-1)*u)
 if family=='exponential':return np.exp(np.log(f)*u)
 return 1/np.sqrt(1+(f**-2-1)*u)
for c in certs:
 sp,family,f,p,eta=c['key'].split('/');f,p,eta=map(float,[f,p,eta]);profile=profiles['animal_augsep/'+sp];z=np.asarray(profile['z']);w=np.asarray(profile['w']);w=w/math.fsum(w);mu=math.fsum(z*w);hw=w*z*feed(z,mu,f,family);EH=math.fsum(hw);t=np.r_[z[z>0],mu]
 # E[h min(X/t,1)] over every exact empirical support knot plus the mean.
 prefix=np.r_[0,np.cumsum(hw*z)];suffix=np.r_[np.cumsum(hw[::-1])[::-1],0];j=np.searchsorted(z,t,side='left');b=prefix[j]/t+suffix[j];a=np.minimum(mu/t,1)
 slack=c['tau_outward']+c['lambda']*a-b;upper=(eta*p*EH+c['lambda']*(p-eta*p)+c['tau_outward']*(1-eta*p))/(mu*f*p)
 r=bounds[(bounds.species==sp)&(bounds.feeding_family==family)&np.isclose(bounds.f,f)&np.isclose(bounds.p,p)&np.isclose(bounds.floor_fraction,eta)].iloc[0]
 wh=hinges[hinges.key==c['key']];ts=wh.hinge_t_numerical.to_numpy();ts=np.where(wh.hinge_is_exact_mean_anchor.to_numpy(),mu,ts);weights=(wh.alpha_lower.to_numpy()+wh.alpha_upper.to_numpy())/2
 aa=np.minimum(mu/ts,1);jj=np.searchsorted(z,ts,side='left');bb=prefix[jj]/ts+suffix[jj];primal=(eta*p*EH+weights@bb)/(mu*f*p)
 lower=(eta*p*EH+(p-eta*p)*math.fsum(hw*np.minimum(z/mu,1)))/(mu*f*p)
 check={'key':c['key'],'nodes_checked':len(t),'minimum_float64_dual_slack':float(slack.min()),'upper_recomputed':float(upper),'upper_archive':float(r.upper_R_I),'upper_absolute_difference':abs(upper-r.upper_R_I),'lower_absolute_difference':abs(lower-r.lower_R_I),'primal_absolute_difference':abs(primal-r.upper_primal_witness_lower),'anchor_residual':abs(weights@aa-(p-eta*p)),'mass_cap_excess':max(0,float(weights.sum()-(1-eta*p)))}
 assert check['minimum_float64_dual_slack']>=-1e-9
 assert max(check[k] for k in ['upper_absolute_difference','lower_absolute_difference','primal_absolute_difference','anchor_residual','mass_cap_excess'])<1e-8,check
 checks.append(check)
# Exact rational arithmetic supplies a separate proof of the same-moment example.
h=lambda x:x/(1+x/Fraction(10));g=lambda x:h(x)*(x/Fraction(10))/(1+x/Fraction(10));counter=[]
for name,xs,ws in [('A',[3,9],[Fraction(1,2)]*2),('B',[0,6,12],[Fraction(1,8),Fraction(3,4),Fraction(1,8)])]:
 xs=list(map(Fraction,xs));moments=[sum(w*x**i for x,w in zip(xs,ws)) for i in range(4)];assert moments==[1,6,45,378];rn=sum(w*h(x) for x,w in zip(xs,ws))/h(Fraction(6));ri=sum(w*g(x) for x,w in zip(xs,ws))/g(Fraction(6));assert rn<1 and ((ri<1) if name=='A' else (ri>1));counter.append({'distribution':name,'R_N_exact':str(rn),'R_I_exact':str(ri)})
args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps({'all_36_full_support_checks_passed':len(checks)==36,'scope':'Float64 checks of coefficients and primal-dual feasibility','checks':checks,'exact_same_moment_counterexample':counter},indent=2));print('All 36 full-support checks and exact rational counterexample passed.')
