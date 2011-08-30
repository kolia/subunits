from   time  import time
from   numpy import dot, minimum, sqrt, zeros, arange, floor
from   numpy import ones, zeros_like, concatenate, sum, newaxis
import numpy.random   as R

from IRLS import IRLS

import sparse_group_lasso as sgl
reload(sgl)

from IPython.Debugger import Tracer; debug_here = Tracer()


def test( n=30 , m=50 , s=8 , r=1 ):
    P = R.randn( n, m )
    x = zeros( (m , r) )
    o = R.permutation(arange(minimum(s,m)))
    x[o[:s],:] = R.randn(s,r)
    y = dot(P,x)
    true_x = [x[s*i:minimum(s*(i+1),P.shape[1])] for i in range(floor(m/s))]
    truex_v = concatenate([tx.squeeze() for tx in true_x])
    def setNone(tx):
        if sum(abs(tx))<1e-10: return None
        else: return tx.squeeze()
    true_x = [setNone(tx) for tx in true_x]
    
    start = time()
    irlsx , irlsw = IRLS(y,P,lam=0.,ftol=1e-7,maxiter=5000,disp_every=1)
    IRLS_time = time() - start
    irlsx[abs(irlsx)<1e-5] = 0
    
    start = time()
    predictors    = [P[:,s*i:minimum(s*(i+1),P.shape[1])] for i in range(floor(m/s))]
    group_weights = [5. for _ in predictors]
    weights       = [5.*ones(p.shape[1]) for p in predictors]
    r,coeffs  = sgl.initialize_group_lasso(predictors, y)
    print r
    for iteration in range(1000):
        old_coeffs = [c[0] for c in coeffs]
        sgl.group_lasso_step(predictors, group_weights, weights, r, coeffs)
        print 'objective: ',sgl.objective(group_weights, weights, r, coeffs),
        print '  nonzero groups: ', len(filter(lambda c : c[0] is not None , coeffs)),
        print '  nonzeros: ', sum( [sum(cc[0]!=0 for cc in 
                                    filter(lambda c : c[0] is not None , coeffs))])
        dc = [sum(abs(oc-c[0])) for oc,c in 
                       filter(lambda c: c[0] is not None and c[1][0] is not None, \
                       zip(old_coeffs,coeffs))]
        if iteration>20 and sum([abs(dcc) for dcc in dc])<1e-5: break
    finish = time()
    print iteration,'Iterations of sparse group LASSO in ',finish-start,' seconds for n,m: ',n,m
    print 'true x '   ,true_x
    print 'infered x ',coeffs

    irls_x = [irlsx[s*i:minimum(s*(i+1),P.shape[1])] for i in range(floor(m/s))]
    irls_x = [setNone(tx.squeeze()) for tx in irlsx]
    print 'IRLS x', irls_x

    def inflate(c,w):
        if c[0] is None: return zeros_like(w)
        else: return c[0]
    coeff_v = concatenate([inflate(c,w) for c,w in zip(coeffs,weights)])
    print 'relative error: ',sum(abs(truex_v-coeff_v)),sum(abs(truex_v)),
    print sum(abs(truex_v-coeff_v))/sum(abs(truex_v))*100,'%'

    coeff_v = coeff_v[:,newaxis]
    xx = x[:coeff_v.size]
    print 'SGL  Errors: ', abs(coeff_v[abs(coeff_v-xx)>1e-4]) - abs(xx[abs(coeff_v-xx)>1e-4])
    print 'IRLS Errors: ', abs(irlsx[abs(irlsx-x)>1e-4]) - abs(x[abs(irlsx-x)>1e-4])
    print 'IRLS ran for ',IRLS_time,' seconds for n,m: ',n,m

    return coeff_v,truex_v