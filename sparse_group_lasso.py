from   time  import time
from   numpy import dot, minimum, maximum, sqrt, sign, zeros, arange, floor
from   numpy import ones, ones_like, array
import numpy.random   as R

from IPython.Debugger import Tracer; debug_here = Tracer()

def newton_lasso(pp,pr,coeffL2,gw,w,c):
    sq = sqrt(coeffL2+c**2)
    Lp = -pr + pp*c + gw*c/sq + w*sign(c)
    Lpp=  pp + gw*(1/sq - c**2/sq**3)
#    print 'Lp ',Lp,'   Lpp ',Lpp,'   c ',c
    return -Lp/Lpp

def optimize_coeffs(predictor,group_weight,weight,rr,coeff):
    # This is ammenable to storing sum(predictor**2) and 
    #             keeping track of sum(predictor*residual) only
    coeffL2 = [sum(coeff[0]**2)]
    PP = sum(predictor*predictor,axis=0)
    old_coeff = coeff[0]
    def objective(old_c,coeffL2,z,w,c): 
        return 0.5*sum((rr[0]-z*(c-old_c))**2) + group_weight*sqrt(coeffL2+c**2) - group_weight*sqrt(coeffL2+old_c**2) + sum(w*abs(c)) - sum(w*abs(old_c))
    def best_c(coeffL2,pp,z,c,w):
        old_c = c
        coeffL2[0] -= c**2
        pr = dot(z,rr[0]) + pp*c
        if abs(pr) < w:
            c = 0.
        else:
            counter = 0
            while 1:
#                print 'obj pre: ', objective(old_c,coeffL2[0],z,w,c),
                dc    = newton_lasso(pp,pr,coeffL2,group_weight,w,c)
                c    += dc
                counter += 1
                if abs(dc)<1e-10 or counter>10: break
#                print '  obj post: ', objective(old_c,coeffL2[0],z,w,c)
            c = c.squeeze()
        coeffL2[0] = coeffL2[0] + c**2
        rr[0] += (old_c-c)*z
        return c

    coeff[0] = array([best_c(coeffL2,pp,z,c,w) for pp,z,c,w 
                      in zip( PP, predictor.T, coeff[0], weight )])
#    rr[0] += dot( predictor, old_coeff-coeff[0] )

def sparsify_group(predictor, group_weight, weight, r, coeff):
    if coeff[0] is None:
        residual = r[0]
    else:
        residual = r[0] + dot(predictor,coeff[0])
    a = dot(predictor.T,residual)
    t = maximum( minimum( a / weight , 1 ) , -1 )
    J = sum( (a - weight * t)**2 ) / group_weight**2
    if J<=1:
        r[0]     = residual
        coeff[0] = None
    elif coeff[0] is None:
        coeff[0] = 1e-8*ones_like(weight)

def initialize_group_lasso(predictors, y, coeffs = None):
    if coeffs is None: coeffs = [[None] for _ in predictors]
    def dotc(p,c):
        if c[0] is None: return 0.
        else:            return dot(p,c)
    N = len(predictors)
    rr = [sum([y/N - dotc(predictor,c) 
          for c,predictor in  zip(coeffs,predictors)],axis=0).squeeze()]
    return rr, coeffs

def group_lasso_step(predictors, group_weights, weights, r, coeffs):
    for predictor , group_weight , weight , coeff  in \
    zip(predictors, group_weights, weights, coeffs ):
        sparsify_group (predictor,group_weight,weight,r,coeff)
        if coeff[0] is not None:
            optimize_coeffs(predictor,group_weight,weight,r,coeff)

def test( n=10 , m=300 , s=25 , r=1 ):
    P = R.randn( n, m )
    x = zeros( (m , r) )
    o = R.permutation(arange(minimum(s*3,m)))
    x[o[:s],:] = R.randn(s,r)
    y = dot(P,x)
    start = time()
    predictors    = [P[:,s*i:minimum(s*(i+1),P.shape[1])] for i in range(floor(m/s))]
    group_weights = [1 for _ in predictors]
    weights       = [ones(p.shape[1]) for p in predictors]
    r,coeffs  = initialize_group_lasso(predictors, y, coeffs = None)
    def objective():
        return 0.5*sum(r[0]**2) + \
        sum([gw*sqrt(sum(c**2)) for gw,c in zip(group_weights,coeffs[0])]) + \
        sum([sum(w*abs(c))      for w ,c in zip(      weights,coeffs[0])])
    print r
    for _ in range(100):
        old_coeffs = [c[0] for c in coeffs]
        group_lasso_step(predictors, group_weights, weights, r, coeffs)
        print 'objective: ',objective(),'  dc: ', [sum(abs(oc-c[0])) for oc,c in 
                       filter(lambda c: c[0] is not None and c[1][0] is not None, zip(old_coeffs,coeffs))]
    finish = time()
    print 'Sparse group LASSO ran for ',finish-start,' seconds for n,m: ',n,m