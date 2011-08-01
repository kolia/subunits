from   numpy        import dot, eye, sum, ones, log, diag, zeros, arange, abs
from   numpy.linalg import slogdet
import scipy.linalg as la
import numpy.random   as R

from time import time

from IPython.Debugger import Tracer; debug_here = Tracer()

def IRLS_step(y,P,x,w,lam):
    P_invW = P/w
    P_invW_P = dot(P,P_invW.T)
    x      = dot( P_invW.T , la.solve( lam*eye(y.size) + P_invW_P , y))
    w      = 1/(x**2 + 1/w - sum( dot( P_invW.T , la.pinv(P_invW_P) ) * P_invW.T , axis=1) )
    return x,w

def ARD(y,P,w):
    S = eye(y.size) + dot(P,(P/w).T)
    s,ldet = slogdet(S)
    return sum(ldet)  + dot(y,dot(la.inv(S),y))
#    L = la.cholesky(eye(y.size) + dot(P,P.T/w))
#    b = la.solve_triangular(L,y)
#    return 2*sum(log(diag(L)))  + dot(y,la.solve_triangular(L.T,b))

def IRLS(y,P,x=0,disp=0,lam=0,maxiter=200):
    w = ones(P.shape[1])
    for i in range(maxiter):
        x,w = IRLS_step(y,P,x,w,lam)
        if disp:
            print 'Iteration ',i,' ARD objective: ',ARD(y,P,w)
    return x,w

def test( n=100 , m=1000 , s=10 ):
    P = R.randn( n, m )
    x = zeros( m )
    o = R.permutation(arange(m))
    x[o[:s]] = R.randn(s)
    y = dot(P,x)
    start = time()
    irlsx , w = IRLS(y,P,disp=0)
    finish = time()
#    print 'w: ',w
#    print 'true x: ', x
    irlsx[abs(irlsx)<1e-7] = 0
#    print 'IRLS x: ', irlsx
    print 'Errors: ', irlsx[abs(irlsx-x)>1e-4] - x[abs(irlsx-x)>1e-4]
    print 'IRLS ran for ',finish-start,' seconds'