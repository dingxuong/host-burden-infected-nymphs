"""Fit annual infection models and integrate infected-nymph density."""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import ndtr, expit, logit, betainc, gammaln, betaln, digamma

GRID = np.linspace(-4.8, 4.8, 241)
EDGES = np.r_[-np.inf, (GRID[:-1] + GRID[1:]) / 2, np.inf]
TEMP = {'iid_tight': (0., .15)}
FEATURES = {'chip_pressure': ['log_chip']}
THETA = np.tanh(np.linspace(-1.5, 1.5, 13))
TRANSITIONS = {}

def configure(data, output):
    global DATA, OUT
    DATA, OUT = data, output
    TRANSITIONS.clear()

def observation(k,n,eta,phi):
    p = np.clip(expit(eta),1e-12,1-1e-12)
    choose = gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
    if np.isinf(phi):
        ll = choose+k*np.log(p)+(n-k)*np.log1p(-p)
        deriv = k-n*p
    else:
        a=p*phi; b=(1-p)*phi
        ll=choose+betaln(k+a,n-k+b)-betaln(a,b)
        deriv=p*(1-p)*phi*(digamma(k+a)-digamma(n-k+b)-digamma(a)+digamma(b))
    return ll,deriv

def name(c):return '_'.join(map(str,c))

def kernel(rho,q,theta=0.,g=None,weight=None):
    if g is None:g=np.array([0.]);weight=np.array([1.])
    sd=q*np.sqrt(1-theta*theta)
    center=rho*GRID[None,:]+q*theta*np.asarray(g)[:,None]
    T=np.zeros((len(GRID),len(GRID)))
    for start in range(0,len(g),32):
        loc=center[start:start+32]
        cdf=ndtr((EDGES[:,None,None]-loc[None,:,:])/sd)
        T+=np.tensordot(np.diff(cdf,axis=0),weight[start:start+32],axes=([1],[0]))
    if abs(T.sum(0)-1).max()>1e-10:raise ValueError('Conditional kernel normalization')
    return T

def transition(year, temporal, index):
    if index != len(THETA) // 2:
        raise ValueError('These models use independent density and infection innovations')
    key = (year, temporal, index)
    if key not in TRANSITIONS:
        rho, q = TEMP[temporal]
        TRANSITIONS[key] = kernel(rho, q)
    return TRANSITIONS[key]

def filtering(beta,groups,temporal,index,phi):
    w=np.exp(-.5*(GRID/.5)**2);w/=w.sum();dw=np.zeros((len(GRID),len(beta)))
    total=0.;grad=np.zeros(len(beta));history=[]
    for idx,(year,X,k,n) in enumerate(groups):
        if idx:
            for step in range(groups[idx-1][0]+1,year+1):
                T=transition(step,temporal,index);w=T@w;dw=T@dw
        ll,deriv=observation(k[None,:],n[None,:],GRID[:,None]+(X@beta)[None,:],phi)
        ell=ll.sum(1);mx=ell.max();likelihood=np.exp(ell-mx);un=w*likelihood;norm=un.sum()
        dll=deriv@X;dun=likelihood[:,None]*dw+un[:,None]*dll;dnorm=dun.sum(0)
        grad+=dnorm/norm;dw=dun/norm-un[:,None]*dnorm[None,:]/norm**2
        w=un/norm;total+=np.log(norm)+mx
        history.append({'year':year,'mean':float(w@GRID),'boundary_mass':float(w[0]+w[-1])})
    return total,grad,w,history

def fit(year,c,index,initial=None):
    tr=DATA[DATA.outcome_year<year].copy()
    te=DATA[DATA.outcome_year.eq(year)].sort_values('Plot').copy()
    cols=FEATURES[c[0]]
    for f in [tr,te]:
        fm=f.mouse*f.burden_m;fc=f.chip*f.burden_c
        f['log_total']=np.log(fm+fc);f['log_ratio']=np.log(fc/fm)
        f['log_chip']=np.log(fc);f['log_mouse']=np.log(fm)
    m=tr[cols].mean().to_numpy();sd=np.maximum(tr[cols].std(ddof=0).to_numpy(),1e-6)
    X=np.column_stack([np.ones(len(tr)),(tr[cols].to_numpy()-m)/sd])
    Z=np.column_stack([np.ones(len(te)),(te[cols].to_numpy()-m)/sd]);prior=np.array([5.]+[c[3]]*len(cols))
    groups=[]
    for y in sorted(tr.outcome_year.unique()):
        ix=np.flatnonzero(tr.outcome_year.to_numpy()==y)
        groups.append((int(y),X[ix],tr.k.to_numpy()[ix],tr.n.to_numpy()[ix]))
    def objective(b):
        ll,g,_,_=filtering(b,groups,c[1],index,c[2])
        return -ll+.5*np.sum((b/prior)**2),-g+b/prior**2
    if initial is None:
        initial=np.zeros(X.shape[1]);p=(tr.k.sum()+.5)/(tr.n.sum()+1);initial[0]=np.log(p/(1-p))
    opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'ftol':1e-11,'gtol':1e-6,'maxiter':600,'maxls':60})
    if not opt.success:raise RuntimeError((year,c,index,opt.message))
    _,_,w,history=filtering(opt.x,groups,c[1],index,c[2])
    rho,q=TEMP[c[1]];forecast=kernel(rho,q)@w
    # Laplace coefficient evidence gives a normalized correlation comparison
    # *within the same feature/dynamics/dispersion/prior configuration*.
    step=1e-4;H=np.empty((len(opt.x),len(opt.x)))
    for j in range(len(opt.x)):
        delta=np.zeros(len(opt.x));delta[j]=step
        H[:,j]=(objective(opt.x+delta)[1]-objective(opt.x-delta)[1])/(2*step)
    H=(H+H.T)/2;ev=np.linalg.eigvalsh(H)
    if ev.min()<=0:raise ValueError(('Laplace curvature',year,c,index,ev))
    logevidence=-float(opt.fun)-np.log(prior).sum()-.5*np.linalg.slogdet(H)[1]
    np.savez_compressed(OUT/f'fit_{year}_{name(c)}_z{index}.npz',offset=(Z@opt.x)[None,:],
        beta=opt.x,state_past=w,state_weight=forecast[None,:],slope=np.zeros(1),beta_weight=np.ones(1),phi=c[2],
        theta=THETA[index],persistence=rho,innovation=q,log_evidence=logevidence)
    return opt.x,{'year':year,'candidate':name(c),'correlation_index':index,'theta':float(THETA[index]),
        'training_end':int(tr.outcome_year.max()),'rows':len(tr),'columns':cols,'mean':m.tolist(),'sd':sd.tolist(),
        'beta':opt.x.tolist(),'log_evidence':logevidence,'Laplace_min_eigenvalue':float(ev.min()),
        'gradient_max':float(abs(objective(opt.x)[1]).max()),'forecast_boundary_mass':float(forecast[0]+forecast[-1]),
        'filtered_history':history,'converged':bool(opt.success)}

def cdf_and_atoms(pp, j):
    atoms, atom_weights, continuous = [], [], []
    for a in pp:
        if a['kind'] == 'source':
            continuous.append(('normal', a['eta'][:, j], a['tau'], np.full(len(a['tau']), a['mass'] / len(a['tau']))))
        else:
            if np.any(a['slope'] != 0) or len(a['offset']) != 1:
                raise ValueError('Integral requires MAP coefficients and independent density.')
            prob = expit(a['offset'][0, j] + GRID)
            w = a['state_weight'][0] * a['mass']
            if np.isinf(a['phi']):
                atoms.extend(prob)
                atom_weights.extend(w)
            else:
                continuous.append(('beta', prob * a['phi'], (1 - prob) * a['phi'], w))
    atoms = np.asarray(atoms)
    atom_weights = np.asarray(atom_weights)

    def ccdf(x):
        x = np.atleast_1d(x)
        value = np.zeros(len(x))
        # Chunking limits intermediate arrays for source posterior draws.
        for lo in range(0, len(x), 256):
            xx = x[lo:lo + 256]
            for kind, p1, p2, w in continuous:
                if kind == 'normal':
                    value[lo:lo + 256] += ndtr((logit(xx)[None, :] - p1[:, None]) / p2[:, None]).T @ w
                else:
                    value[lo:lo + 256] += betainc(p1[:, None], p2[:, None], xx[None, :]).T @ w
        return value

    def total_cdf(x):
        return float(ccdf(x)[0] + atom_weights[atoms <= x].sum())

    lo, hi = 0., 1.
    for _ in range(55):
        mid = (lo + hi) / 2
        if total_cdf(mid) >= .5:
            hi = mid
        else:
            lo = mid
    return ccdf, atoms, atom_weights, (lo + hi) / 2, sum(c[3].sum() for c in continuous)

def risk_median(ccdf, atoms, aw, mass, mu, sd, dw, nodes):
    edges_logit = np.linspace(-14, 14, nodes)
    edges = expit(edges_logit)
    c = ccdf(edges)
    weights = np.diff(c)
    if np.any(weights < -1e-12):
        raise ValueError('Nonmonotone probability CDF.')
    weights = np.maximum(weights, 0)
    values = expit((edges_logit[1:] + edges_logit[:-1]) / 2)
    prob = np.r_[atoms, edges[0], values, edges[-1]]
    pw = np.r_[aw, c[0], weights, max(mass - c[-1], 0)]
    mask = pw > 1e-15
    prob, pw = prob[mask], pw[mask]
    if abs(pw.sum() - 1) > 1e-10:
        raise ValueError(('Probability normalization', pw.sum()))
    logp = np.log(prob)
    shifts = mu[:, None] + logp[None, :]
    joint_weights = dw[:, None] * pw[None, :]

    def cdf(logr):
        return np.sum(ndtr((logr - shifts) / sd[:, None]) * joint_weights) - .5

    med = brentq(cdf, float(mu.min() + logp.min() - 10 * sd.max()),
                 float(mu.max() + logp.max() + 10 * sd.max()), xtol=1e-10)
    return float(np.exp(med)), float(c[0] + max(mass - c[-1], 0))
