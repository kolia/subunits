from   numpy        import dot, eye, sum, ones, log, diag, zeros, arange, abs
from   numpy.linalg import slogdet
import scipy.linalg as la
import numpy.random   as R

from time import time

from IPython.Debugger import Tracer; debug_here = Tracer()

def IRLS_step(y,P,x,iw,lam):
    P_iW = P*iw
    P_iW_P = dot(P,P_iW.T)
    x      = dot( P_iW.T , la.solve( lam*eye(y.size) + P_iW_P , y))
    iw     = x**2 + iw - sum( dot( P_iW.T , la.pinv(P_iW_P) ) * P_iW.T , axis=1)
    return x,iw

def sIRLS_step(y,P,x,iw,lam):
    nonzero = iw>1e-5
    sP = P[:,nonzero]
    siw = iw[nonzero]
    sP_iW = sP*siw
    sP_iW_sP = dot(sP,sP_iW.T)
    sx = dot( sP_iW.T , la.solve( lam*eye(y.size) + sP_iW_sP , y))
    siw = sx**2 + siw - sum( dot( sP_iW.T , la.pinv(sP_iW_sP) ) * sP_iW.T , axis=1)
    x  = zeros(P.shape[1])
    x[nonzero]  = sx
    iw = zeros(P.shape[1])
    iw[nonzero] = siw
    return x,iw

def ARD(y,P,iw):
    S = eye(y.size) + dot(P,(P*iw).T)
    s,ldet = slogdet(S)
    return sum(ldet)  + dot(y,dot(la.inv(S),y))
#    L = la.cholesky(eye(y.size) + dot(P,P.T/w))
#    b = la.solve_triangular(L,y)
#    return 2*sum(log(diag(L)))  + dot(y,la.solve_triangular(L.T,b))

def IRLS(y,P,x=0,disp=0,lam=0,maxiter=200):
    iw = 1e-4 * ones(P.shape[1])
    for i in range(maxiter):
        old_x = x
        x,iw = sIRLS_step(y,P,x,iw,lam)
        if disp:
            print 'Iteration ',i,' nonzeros: ',sum(iw>1e-6)
#            print 'Iteration ',i,' ARD objective: ',ARD(y,P,iw)
        if sum(abs(x-old_x))<1e-8: break
    return x,iw

def test( n=200 , m=5000 , s=10 ):
    P = R.randn( n, m )
    x = zeros( m )
    o = R.permutation(arange(m))
    x[o[:s]] = R.randn(s)
    y = dot(P,x)
    start = time()
    irlsx , w = IRLS(y,P,disp=1)
    finish = time()
#    print 'w: ',w
#    print 'true x: ', x
    irlsx[abs(irlsx)<1e-7] = 0
#    print 'IRLS x: ', irlsx
    print 'Errors: ', irlsx[abs(irlsx-x)>1e-4] - x[abs(irlsx-x)>1e-4]
    print 'IRLS ran for ',finish-start,' seconds for n,m: ',n,m