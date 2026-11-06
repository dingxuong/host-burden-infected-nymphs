'Calculate conditional output bounds and verify the same-moment example.'
from pathlib import Path
from fractions import Fraction
import csv, gzip, hashlib, json, math, collections
import numpy as np
import scipy
from scipy.optimize import linprog, brentq
from mpmath import iv

BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'empirical_shape_public_v1'
SHAPE=PRIOR/'normalized_observed_shapes.csv.gz'
FAMILIES=['reciprocal','exponential','inverse_sqrt']
EXAMPLES=[(.8,.2),(.2,.8),(.2,.2)]
NINE=[(f,p) for f in [.2,.5,.8] for p in [.2,.5,.8]]
VERSIONS=['animal_augsep','capture_clean_augsep','capture_source_augsep','animal_julaugsep']
iv.dps=50

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=[SHAPE]
input_hashes={str(p):sha(p) for p in frozen}

def dump_csv(name,rows):
    with (BASE/name).open('w',encoding='utf-8-sig',newline='') as h:
        out=csv.DictWriter(h,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)

def feeding(sf,z,f):
    if sf=='reciprocal':return 1/(1+(1/f-1)*z)
    if sf=='exponential':return np.exp(np.log(f)*z)
    return 1/np.sqrt(1+(f**-2-1)*z)

def infect(qf,z,p,eta):
    q0=eta*p
    if qf=='rational':
        b=(p-q0)/(1-p);return q0+(1-q0)*b*z/(1+b*z)
    b=-math.log((1-p)/(1-q0));return q0+(1-q0)*(-np.expm1(-b*z))

raw=collections.defaultdict(lambda:[[],[]])
with gzip.open(SHAPE,'rt',encoding='utf-8-sig') as h:
    for r in csv.DictReader(h):
        d=raw[(r['version'],r['species'])]
        d[0].append(float(r['normalized_z']));d[1].append(float(r['pooled_weight']))
profiles={};profile_records={}
for key,(z,w) in raw.items():
    z=np.asarray(z);w=np.asarray(w)
    nodes,inverse=np.unique(z,return_inverse=True)
    weights=np.bincount(inverse,weights=w)
    weights=weights/math.fsum(weights)
    mu=math.fsum(nodes*weights)
    assert abs(mu-1)<1e-12
    profiles[key]=(nodes,weights,mu)
    profile_records['/'.join(key)]={'z':nodes.tolist(),'w':weights.tolist(),'numerical_mean':mu}
(BASE/'numeric_profile_inputs.json').write_text(json.dumps(profile_records,separators=(',',':')))

# Exact synthetic counterexample, raw moments through degree three.
counter=[]
a=b=Fraction(1,10);mu=Fraction(6)
h=lambda x:x/(1+a*x)
g=lambda x:h(x)*b*x/(1+b*x)
synthetic={'A':[(Fraction(3),Fraction(1,2)),(Fraction(9),Fraction(1,2))],
           'B':[(Fraction(0),Fraction(1,8)),(Fraction(6),Fraction(3,4)),(Fraction(12),Fraction(1,8))]}
for name,rows in synthetic.items():
    moments=[sum(w*x**j for x,w in rows) for j in range(4)]
    assert moments==[1,6,45,378]
    rn=sum(w*h(x) for x,w in rows)/h(mu)
    ri=sum(w*g(x) for x,w in rows)/g(mu)
    assert rn<1 and (ri<1 if name=='A' else ri>1)
    counter.append({'distribution':name,'probability_mass':' '.join(f'{x}:{w}' for x,w in rows),
        'mean':6,'variance':9,'third_central_moment':0,
        'raw_moments_0_to_3':' '.join(map(str,moments)),
        'R_N_exact':str(rn),'R_I_exact':str(ri),'R_N':float(rn),'R_I':float(ri)})
dump_csv('exact_same_moments_counterexample.csv',counter)

def stoploss(z,w,t):
    ps=np.r_[np.cumsum(w[::-1])[::-1],0.]
    ms=np.r_[np.cumsum((w*z)[::-1])[::-1],0.]
    i=np.searchsorted(z,t,side='right')
    return ms[i]-t*ps[i]

# A piecewise-linear difference needs checking only at its complete knot set.
comparisons=[(('animal_augsep','PL'),('animal_augsep','TS'))]
comparisons += [(('animal_augsep',sp),(v,sp)) for sp in ['PL','TS'] for v in VERSIONS[1:]]
order_rows=[]
for left,right in comparisons:
    z,w,mu=profiles[left];zz,ww,mm=profiles[right]
    z=z/mu;zz=zz/mm
    t=np.unique(np.r_[0.,z,zz])
    delta=stoploss(zz,ww,t)-stoploss(z,w,t)
    lo,hi=float(delta.min()),float(delta.max());tol=1e-10
    if lo>=-tol and hi>tol:status='left_le_cx_right_within_tolerance'
    elif hi<=tol and lo<-tol:status='right_le_cx_left_within_tolerance'
    elif lo<-tol and hi>tol:status='incomparable'
    else:status='indistinguishable_within_tolerance'
    order_rows.append({'left':'/'.join(left),'right':'/'.join(right),'status':status,
        'min_stoploss_difference':lo,'max_stoploss_difference':hi,'knots':len(t),'absolute_tolerance':tol})
dump_csv('empirical_convex_order_checks.csv',order_rows)

# Global curvature decomposition using exact interval-antiderivative identities.
def curvature_check(z,w,f,p,label):
    mu=math.fsum(z*w);a=1/f-1;b=p/(1-p)
    def G(x):
        t=x/mu;return mu*b*t*t/((1+a*t)*(1+b*t))
    def Gprime(x):
        t=x/mu;return b*(2*t+(a+b)*t*t)/((1+a*t)**2*(1+b*t)**2)
    def poly(t):return 1-3*a*b*t*t-a*b*(a+b)*t**3
    right=max(1.,float(z.max()/mu))
    while poly(right)>0:right*=2
    root=mu*brentq(poly,0.,right,xtol=1e-13)
    knots=np.unique(np.r_[0.,z,mu,root if root<float(z.max()) else []])
    ps=np.r_[np.cumsum(w[::-1])[::-1],0.]
    ms=np.r_[np.cumsum((w*z)[::-1])[::-1],0.]
    areas_pos=[];areas_neg=[]
    for l,r in zip(knots[:-1],knots[1:]):
        mid=(l+r)/2;i=np.searchsorted(z,mid,side='right')
        d=ps[i]-(1. if mid<mu else 0.)
        c=ms[i]-(mu if mid<mu else 0.)
        kl,kr=c-d*l,c-d*r
        area=kr*Gprime(r)-kl*Gprime(l)+d*(G(r)-G(l))
        (areas_pos if mid<root else areas_neg).append(area)
    pos=math.fsum(areas_pos);neg=-math.fsum(areas_neg)
    direct=math.fsum(w*G(z))-G(mu);integrated=pos-neg
    assert abs(direct-integrated)<1e-8 and pos>-1e-9 and neg>-1e-9
    return {'profile':label,'f':f,'p':p,'curvature_root':root,
        'positive_curvature_contribution':pos/G(mu),'negative_curvature_loss':neg/G(mu),
        'direct_R_I_minus_1':direct/G(mu),'integral_R_I_minus_1':integrated/G(mu),
        'absolute_identity_error':abs(direct-integrated),'full_support_max':float(z.max())}
curvature=[]
for name,rows in synthetic.items():
    zz=np.array([float(x) for x,w in rows]);ww=np.array([float(w) for x,w in rows])
    curvature.append(curvature_check(zz,ww,.625,.375,'synthetic_'+name))
for sp in ['PL','TS']:
    z,w,mu=profiles[('animal_augsep',sp)]
    for f,p in NINE:curvature.append(curvature_check(z,w,f,p,'animal_augsep/'+sp))
dump_csv('global_curvature_decomposition.csv',curvature)

# C0 sharp bounds with pointwise attainable monotone extremizers.
c0=[]
for sp in ['PL','TS']:
    z,w,mu=profiles[('animal_augsep',sp)]
    for sf in FAMILIES:
        for f,p in NINE:
            hh=z*feeding(sf,z/mu,f);hw=hh*w;eh=math.fsum(hw);rn=eh/(mu*f)
            for eta in [0.,.5]:
                q0=eta*p
                lower=(q0*math.fsum(hw[z<mu])+p*math.fsum(hw[z>=mu]))/(mu*f*p)
                upper=(p*math.fsum(hw[(z>0)&(z<=mu)])+math.fsum(hw[z>mu]))/(mu*f*p)
                assert lower<=rn+1e-12
                originals=[math.fsum(hw*infect(qf,z/mu,p,eta))/(mu*f*p) for qf in ['rational','exponential']]
                assert all(lower-1e-10<=r<=upper+1e-10 for r in originals)
                c0.append({'species':sp,'feeding_family':sf,'f':f,'p':p,'floor_fraction':eta,
                    'R_N':rn,'lower_R_I':lower,'upper_R_I':upper,
                    'direction':'guaranteed_up' if lower>1.01 else 'guaranteed_down' if upper<.99 else 'not_identified',
                    'original_R_I_min':min(originals),'original_R_I_max':max(originals)})
dump_csv('C0_monotone_sharp_bounds.csv',c0)

def lower_float(x):return float(np.nextafter(float(x.a),-np.inf))
def upper_float(x):return float(np.nextafter(float(x.b),np.inf))

def min_one(x):
    if x.a>=1:return iv.mpf(1)
    if x.b<=1:return x
    return iv.mpf([x.a,1])

def iv_basis(z,w,sf,f):
    """Interval enclosures for the declared decimal-serialized empirical inputs."""
    zs=[iv.mpf(str(x)) for x in z];ws=[iv.mpf(str(x)) for x in w]
    wsum=sum(ws,iv.mpf(0));ws=[x/wsum for x in ws]
    mu=sum((x*y for x,y in zip(zs,ws)),iv.mpf(0));fi=iv.mpf(str(f))
    def fs(x):
        u=x/mu
        if sf=='reciprocal':return 1/(1+(1/fi-1)*u)
        if sf=='exponential':return iv.exp(iv.ln(fi)*u)
        return 1/iv.sqrt(1+(fi**-2-1)*u)
    hw=[wi*zi*fs(zi) for wi,zi in zip(ws,zs)]
    eh=sum(hw,iv.mpf(0))
    pref=[iv.mpf(0)]
    for x,y in zip(zs,hw):pref.append(pref[-1]+x*y)
    suff=[iv.mpf(0)]*(len(zs)+1)
    for i in range(len(zs)-1,-1,-1):suff[i]=suff[i+1]+hw[i]
    # Include every positive observed knot and an exact mean anchor.
    nodes=[(float(z[i]),zs[i],False,i) for i in range(len(z)) if z[i]>0]
    nodes.append((float(math.fsum(z*w)),mu,True,-1))
    nodes.sort(key=lambda x:x[0])
    aa=[];bb=[]
    for _,t,anchor,i in nodes:
        if anchor:
            aa.append(iv.mpf(1))
            bb.append(sum((hi*min_one(zi/mu) for hi,zi in zip(hw,zs)),iv.mpf(0)))
        else:
            aa.append(min_one(mu/t));bb.append(pref[i]/t+suff[i])
    return nodes,aa,bb,mu,eh,fi

c1=[];witnesses=[];certificates=[]
for sp in ['PL','TS']:
    z,w,mu=profiles[('animal_augsep',sp)]
    for sf in FAMILIES:
        for f in [.2,.8]:
            print(f'Preparing interval coefficients: {sp}, {sf}, f={f}',flush=True)
            nodes,aa,bb,mi,eh,fi=iv_basis(z,w,sf,f)
            a_num=np.array([float(x.mid) for x in aa]);b_num=np.array([float(x.mid) for x in bb])
            anchor_index=next(i for i,n in enumerate(nodes) if n[2])
            for ff,p in EXAMPLES:
                if ff!=f:continue
                for eta in [0.,.5]:
                    pi=iv.mpf(str(p));q0i=iv.mpf(str(eta))*pi
                    targeti=pi-q0i;capi=1-q0i
                    q0=eta*p;target=p-q0;cap=1-q0
                    solution=linprog(-b_num,A_ub=np.ones((1,len(nodes))),b_ub=[cap],
                        A_eq=a_num[None,:],b_eq=[target],bounds=(0,None),method='highs',
                        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
                    assert solution.success,solution.message
                    # SciPy is a minimizer; reverse both marginals for the max dual.
                    lam=-float(solution.eqlin.marginals[0]);lam_i=iv.mpf(str(lam))
                    tau_raw=max(0.,max(upper_float(bj-lam_i*aj) for aj,bj in zip(aa,bb)))
                    tau=float(np.nextafter(tau_raw+1e-12,np.inf));tau_i=iv.mpf(str(tau))
                    slack_min=min(lower_float(tau_i+lam_i*aj-bj) for aj,bj in zip(aa,bb))
                    assert slack_min>=0
                    denominator=mi*fi*pi
                    upper_i=(q0i*eh+lam_i*targeti+tau_i*capi)/denominator
                    lower_i=(q0i*eh+targeti*bb[anchor_index])/denominator
                    rn_i=eh/(mi*fi)
                    # Recover a feasible one- or two-hinge extremizer algebraically.
                    active=np.flatnonzero(solution.x>1e-8)
                    assert 1<=len(active)<=2,(len(active),solution.x[active])
                    if len(active)==1:
                        j=int(active[0]);alphas=[targeti/aa[j]]
                        assert lower_float(alphas[0])>=0 and lower_float(capi-alphas[0])>=0
                    else:
                        j,k=map(int,active)
                        first=(targeti-capi*aa[k])/(aa[j]-aa[k]);alphas=[first,capi-first]
                        assert all(lower_float(v)>=0 for v in alphas)
                    primal_i=(q0i*eh+sum((v*bb[int(j)] for j,v in zip(active,alphas)),iv.mpf(0)))/denominator
                    upper=upper_float(upper_i);lower=lower_float(lower_i);primal=lower_float(primal_i)
                    gap=upper-primal
                    assert -1e-12<=gap<1e-7,gap
                    original=[math.fsum(w*z*feeding(sf,z/mu,f)*infect(qf,z/mu,p,eta))/(mu*f*p)
                              for qf in ['rational','exponential']]
                    assert all(lower-1e-10<=r<=upper+1e-10 for r in original)
                    outer=next(r for r in c0 if r['species']==sp and r['feeding_family']==sf
                               and r['f']==f and r['p']==p and r['floor_fraction']==eta)
                    assert outer['lower_R_I']<=lower+1e-10 and upper<=outer['upper_R_I']+1e-10
                    row={'species':sp,'feeding_family':sf,'f':f,'p':p,'floor_fraction':eta,
                        'R_N_lower':lower_float(rn_i),'R_N_upper':upper_float(rn_i),
                        'lower_R_I':lower,'upper_R_I':upper,'upper_primal_witness_lower':primal,
                        'certified_upper_gap':gap,'dual_slack_lower_min':slack_min,
                        'direction':'guaranteed_up' if lower>1.01 else 'guaranteed_down' if upper<.99 else 'not_identified',
                        'original_R_I_min':min(original),'original_R_I_max':max(original),
                        'basis_knots':len(nodes),'nonzero_hinges':len(active),
                        'solver_anchor_residual':float(abs(a_num@solution.x-target)),
                        'solver_cap_excess':float(max(0.,solution.x.sum()-cap))}
                    c1.append(row)
                    key=f'{sp}/{sf}/{f}/{p}/{eta}'
                    certificates.append({'key':key,'lambda':lam,'tau_outward':tau,
                        'upper_R_I':upper,'dual_slack_lower_min':slack_min,'interval_dps':iv.dps})
                    for j,alpha in zip(active,alphas):
                        node=nodes[int(j)]
                        witnesses.append({'key':key,'hinge_t_numerical':node[0],'hinge_is_exact_mean_anchor':node[2],
                            'alpha_interval':str(alpha),'alpha_lower':lower_float(alpha),'alpha_upper':upper_float(alpha)})
            print(f'Completed: {sp}, {sf}, f={f}',flush=True)
dump_csv('C1_concave_sharp_bounds.csv',c1)
dump_csv('C1_upper_extremizer_hinges.csv',witnesses)
(BASE/'C1_dual_certificates.json').write_text(json.dumps(certificates,indent=2))

for path,d in input_hashes.items():assert sha(Path(path))==d,path
summary={'analysis_type':'Conditional output bounds on the observed burden distributions',
    'input_sha256':input_hashes,'scipy_version':scipy.__version__,'interval_dps':iv.dps,
    'same_moment_counterexample':'exact rational pass',
    'curvature_models':len(curvature),'curvature_max_absolute_error':max(r['absolute_identity_error'] for r in curvature),
    'empirical_convex_order_comparisons':order_rows,
    'C0_models':len(c0),'C0_directions':dict(collections.Counter(r['direction'] for r in c0)),
    'C1_models':len(c1),'C1_directions':dict(collections.Counter(r['direction'] for r in c1)),
    'C1_max_certified_gap':max(r['certified_upper_gap'] for r in c1),
    'C1_min_dual_slack_lower':min(r['dual_slack_lower_min'] for r in c1),
    'C1_original_function_containment':'pass','C1_in_C0_nesting':'pass',
    'frozen_inputs_unchanged':'pass','new_model_fitting':False,
    'bound_meaning':'Minimum and maximum output across the specified response-function class',
    'interval_semantics':'50-decimal interval arithmetic; include decimal input values, normalized weights and mean uncertainty; check every hinge constraint.',
    'stopping_status':'Calculations completed'}
(BASE/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','empirical_convex_order_comparisons']},ensure_ascii=False,indent=2))
