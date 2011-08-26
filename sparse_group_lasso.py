from   time  import time
from   numpy import dot, minimum, maximum, sqrt, sign, zeros, arange, floor
from   numpy import ones, ones_like
import numpy.random   as R

from IPython.Debugger import Tracer; debug_here = Tracer()

def newton_lasso(pp,pr,coeffL2,gw,w,c):
    sq = sqrt(coeffL2+c**2)
    Lp = -pr + pp*c + gw*c/sq + w*sign(c)
    Lpp=  pp + gw*(1/sq - c**2/sq**3)
    print 'Lp ',Lp,'   Lpp ',Lpp,'   c ',c
    return - Lp/Lpp

def optimize_coeffs(predictor,group_weight,weight,rr,coeff):
    # This is ammenable to storing sum(predictor**2) and 
    #             keeping track of sum(predictor*residual) only
    coeffL2 = sum(coeff[0]**2)
    PP = sum(predictor*predictor,axis=0)
    for pp,z,c,w in zip( PP, predictor.T, coeff[0], weight ):
        coeffL2 -= c**2
        pr = dot(z,rr[0]) + pp*c
        if abs(pr) < w:
            c = 0
        else:
            while 1:
                dc    = newton_lasso(pp,pr,coeffL2,group_weight,w,c)
                c    += dc
                pr   += dc * pp
                rr[0] += dc * z
                if abs(dc)<1e-5: break
        coeff[0] = c
        coeffL2 += c**2

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
        if coeff is not None:
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
    r,coeffs = initialize_group_lasso(predictors, y, coeffs = None)
    print r
    for _ in range(100):
        old_coeffs = coeffs
        group_lasso_step(predictors, group_weights, weights, r, coeffs)
        print 'dc: ', [sum(abs(oc[0]-c[0])) for oc,c in 
                       filter(lambda c: c[0][0] is not None, zip(old_coeffs,coeffs))]
    finish = time()
    print 'Sparse group LASSO ran for ',finish-start,' seconds for n,m: ',n,m