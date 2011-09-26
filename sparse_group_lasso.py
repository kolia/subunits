import numpy
from   numpy import dot, minimum, maximum, sqrt, sign, concatenate
from   numpy import ones_like, array, sum, zeros, newaxis

from IPython.Debugger import Tracer; debug_here = Tracer()

def newton_lasso(pp,pr,coeffL2,gw,w,c):
    sq = sqrt(coeffL2+c**2)
    Lp = -pr + pp*c + gw*c/sq + w*sign(c)
    Lpp=  pp + gw*(1/sq - c**2/sq**3)
    return -Lp/Lpp

def optimize_coeffs(predictor,group_weight,weight,rr,coeff):
    # This is ammenable to storing sum(predictor**2) and 
    #             keeping track of sum(predictor*residual) only
    coeffL2 = [sum(coeff[0]**2)]
    PP = sum(predictor*predictor,axis=0)
    def objective(old_c,coeffL2,z,w,c): 
        return 0.5*sum((rr[0]-z*(c-old_c))**2) + \
        group_weight*sqrt(coeffL2+c**2) - \
        group_weight*sqrt(coeffL2+old_c**2) + sum(w*abs(c)) - sum(w*abs(old_c))
    def best_c(coeffL2,pp,z,c,w):
        old_c = c
        coeffL2[0] -= c**2
        pr = dot(z,rr[0]) + pp*c
        if abs(pr) < w:
            c = 0.
        else:
            counter = 0
            while 1:
                old_obj = objective(old_c,coeffL2[0],z,w,c)
                dc    = newton_lasso(pp,pr,coeffL2[0],group_weight,w,c)
                while objective(old_c,coeffL2[0],z,w,c+dc)>old_obj:
                    dc = dc/2
                c    += dc
                counter += 1
                if abs(dc)<1e-10 or counter>20: 
                    break
            c = c.squeeze()
        coeffL2[0] = coeffL2[0] + c**2
        rr[0] += (old_c-c)*z
        return c
    coeff[0] = array([best_c(coeffL2,pp,z,c,w) for pp,z,c,w 
                      in zip( PP, predictor.T, coeff[0], weight )])

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
    r = [y]
    calculate_residual( predictors, r, coeffs )
    return r, coeffs

def calculate_residual( predictors, r, coeffs ):
    def dotc(p,c):
        if c[0] is None: return 0.
        else:            return dot(p,c[0])
    N = len(predictors)
    r[0] = numpy.sum([r[0]/N - dotc(predictor,c) 
                      for c,predictor in  zip(coeffs,predictors)],axis=0).squeeze()
    
def group_lasso_step(predictors, group_weights, weights, r, coeffs):
    for predictor , group_weight , weight , coeff  in \
    zip(predictors, group_weights, weights, coeffs ):
        sparsify_group (predictor,group_weight,weight,r,coeff)
        if coeff[0] is not None:
            optimize_coeffs(predictor,group_weight,weight,r,coeff)

def sparse_group_lasso(predictors, group_weights, weights, r, coeffs, 
                       maxiter=10000, disp_every=0, ftol=1e-9):
    for iteration in range(maxiter):
        old_coeffs = [c[0] for c in coeffs]
        group_lasso_step( predictors, group_weights, weights, r, coeffs )
        if disp_every and not iteration%disp_every:
            print 'objective: ',current_objective(group_weights, weights, r, coeffs),
            print '  nonzero groups: ',len(filter(lambda c: c[0] is not None,coeffs)),
            print '  nonzeros: ', sum( [sum(cc[0]!=0 for cc in 
                                        filter(lambda c: c[0] is not None , coeffs))])
        dc = [sum(abs(oc-c[0])) for oc,c in 
                       filter(lambda c: c[0] is not None and c[1][0] is not None, \
                       zip(old_coeffs,coeffs))]
        if iteration>20 and sum([abs(dcc) for dcc in dc])<ftol: break

def inflate(coeffs):
    for c in coeffs:
        if c[0] is not None: break
    if c[0] is None: return None
    shape = c[0].shape
    def shaped(c,shape):
        if c is None: return zeros(shape)
        return c
    return concatenate( [shaped(c[0],shape)[:,newaxis].T for c in coeffs] )

def current_objective(group_weights, weights, r, coeffs):
    return 0.5*sum(r[0]**2) + \
    sum([gw*sqrt(sum(c[0]**2)) for gw,c in 
         filter(lambda c : c[1][0] is not None , zip(group_weights,coeffs))]) + \
    sum([sum(w*abs(c[0]))      for w ,c in
         filter(lambda c : c[1][0] is not None , zip(      weights,coeffs))])
         
def objective( predictors, group_weights, weights, y, coeffs ):
    r = [y]
    calculate_residual(predictors,r,coeffs)
    return 0.5*sum(r[0]**2) + \
    sum([gw*sqrt(sum(c[0]**2)) for gw,c in 
         filter(lambda c : c[1][0] is not None , zip(group_weights,coeffs))]) + \
    sum([sum(w*abs(c[0]))      for w ,c in
         filter(lambda c : c[1][0] is not None , zip(      weights,coeffs))])