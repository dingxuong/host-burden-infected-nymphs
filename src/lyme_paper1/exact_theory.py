'Exact A/B distributions with matching mean, variance and skewness.'
from fractions import Fraction as F

def counterexample():
    h = lambda x: x / (1 + x / 10)
    g = lambda x: h(x) * (x / 10) / (1 + x / 10)
    results = []
    for label, nodes, weights in [("A", [3, 9], [F(1,2), F(1,2)]),
                                 ("B", [0,6,12], [F(1,8), F(3,4), F(1,8)])]:
        x = list(map(F, nodes))
        mean = sum(w*z for w,z in zip(weights,x))
        variance = sum(w*(z-mean)**2 for w,z in zip(weights,x))
        third = sum(w*(z-mean)**3 for w,z in zip(weights,x))
        rn = sum(w*h(z) for w,z in zip(weights,x)) / h(mean)
        ri = sum(w*g(z) for w,z in zip(weights,x)) / g(mean)
        results.append(dict(distribution=label, mean=str(mean), variance=str(variance),
                            third_central_moment=str(third), R_N_exact=str(rn),
                            R_I_exact=str(ri), R_N=float(rn), R_I=float(ri)))
    return results
