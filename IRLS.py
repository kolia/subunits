from   numpy        import dot, eye, sum, ones, sqrt, zeros, arange, abs
from   numpy        import tensordot, reshape, prod
from   numpy.linalg import slogdet
import scipy.linalg as la
import numpy.random   as R
from time import time
#from IPython.Debugger import Tracer; debug_here = Tracer()

def IRLS_step_matrix(y,P,iw,lam):
    P_iW = P*iw
    P_iW_P = dot(P,P_iW.T)
    C = la.pinv( lam*eye(y.shape[0]) + P_iW_P)
#    x  = dot( P_iW.T , la.solve( lam*eye(y.size) + P_iW_P , y))
    x  = dot( P_iW.T , dot( C , y ))
    iw = sum(x**2,axis=1) + iw - sum( dot( P_iW.T , C ) * P_iW.T , axis=1)
    return x,iw

#def tensorinv(P):
#    N  = prod(P.shape[:P.ndim/2])
#    fP = reshape(P,(N,-1))
#    invfP = la.pinv(fP)
#    return reshape(invfP,P.shape)
#
#def IRLS_step_tensor(y,P,iw,lam):               # y:      45 by 12,  P: 45 by 12 by 200
#    P_iW = P*iw                                 # P_iW:   45 by 12 by 200
#    P_iW_P = tensordot(P,P_iW.T,axes=1)         # P_iW_P: 45 by 12 by 12 by 45
#    C  = tensorinv( lam*reshape(eye(y.size),y.shape) + P_iW_P)
#    x  = tensordot( P_iW.T , tensordot( C , y ))
#    iw = sum(x**2,axis=1) + iw - sum( dot( P_iW.T , C ) * P_iW.T , axis=1)
#    return x,iw

def IRLS(y,P,x=0,disp_every=0,lam=0,maxiter=1000,ftol=1e-6,iw=1e-1,nonzero=1e-3):
    if isinstance( iw , type(float)):
        iw = iw * ones(P.shape[1])
    for i in range(maxiter):
        old_x = x
        x,iw = IRLS_step_matrix(y,P,iw,lam)
        if disp_every and not i%disp_every:
            print 'Iteration ',i,'  nonzero weights:',sum(iw>nonzero), \
                  '  dL1(x): ',sum(abs(x-old_x))
        if sum(abs(x-old_x))<ftol: break
    return x,iw

def sIRLS_step(y,P,x,iw,lam,nonzero):
    P_iW = P*iw
    P_iW_P = dot(P,P_iW.T)
    C   = la.pinv( lam*eye(y.shape[0]) + P_iW_P)
    x   = dot( P_iW.T , dot( C , y ))
    iw  = sum(x**2,axis=1) + iw - sum( dot( P_iW.T , C ) * P_iW.T , axis=1)
    PC  = dot( P.T , C )
    PCP = sum( PC * P.T , axis=1 )
    PCT = sum( PC * sum(y,axis=1) , axis=1 )
    s = PCP - PCP**2/(PCP-1/(iw+1e-15))
    q = PCT - PCP*PCT/(PCP-1/(iw+1e-15))
    old_nonzero = nonzero
    nonzero = q**2 > s
    siw = iw[nonzero]
    if sum(abs(nonzero-old_nonzero)) < 1:
        if sum(iw>1e-5)>=sum(nonzero):
            iw = iw*0.5
        elif sum(iw>1e-5)<sum(nonzero):
            iw = iw ** 0.9
        iw[nonzero] = siw
    return x,iw,nonzero

def sIRLS(y,P,x=0,disp=0,lam=0,maxiter=1000):
    nz = ones(P.shape[1]) > 0
    iw = 1e-4 * ones(P.shape[1])
    for i in range(maxiter):
        old_x = x
        x,iw,nz = sIRLS_step(y,P,x,iw,lam,nz)
        if disp:
            print 'Iteration ',i,' nonzeros: ',sum(nz),'  nziw:',sum(iw>1e-5),\
                  '  dL1(x): ',sum(abs(x-old_x))
        if sum(abs(x-old_x))<1e-4: break
    return x,iw

def test( n=100 , m=3000 , s=25 , r=1 ):
    P = R.randn( n, m )
    x = zeros( (m , r) )
    o = R.permutation(arange(m))
    x[o[:s],:] = R.randn(s,r)
    y = dot(P,x)
    start = time()
    irlsx , w = IRLS(y,P,disp_every=1)
    finish = time()
    irlsx[abs(irlsx)<1e-7] = 0
    print 'Errors: ', irlsx[abs(irlsx-x)>1e-4] - x[abs(irlsx-x)>1e-4]
    print 'IRLS ran for ',finish-start,' seconds for n,m: ',n,m
    
    sstart = time()
    sirlsx , sw = sIRLS(y,P,disp=1)
    sfinish = time()
    irlsx[abs(irlsx)<1e-7] = 0

    print 'Errors: ', irlsx[abs(irlsx-x)>1e-4] - x[abs(irlsx-x)>1e-4]
    print 'IRLS ran for ',finish-start,' seconds for n,m: ',n,m
    print 'Errors: ', sirlsx[abs(sirlsx-x)>1e-4] - x[abs(sirlsx-x)>1e-4]
    print 'sIRLS ran for ',sfinish-sstart,' seconds for n,m: ',n,m
    print 'ARD: ',ARD(y,P,w)
    print 'sARD: ',ARD(y,P,sw)
    

def ARD(y,P,iw):
    S = eye(y.size) + dot(P,(P*iw).T)
    s,ldet = slogdet(S)
    return sum(ldet)  + dot(y.T,dot(la.inv(S),y))
#    L = la.cholesky(eye(y.size) + dot(P,P.T/w))
#    b = la.solve_triangular(L,y)
#    return 2*sum(log(diag(L)))  + dot(y,la.solve_triangular(L.T,b))
    
def s0IRLS_step(y,P,x,iw,lam):
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

def s2IRLS_step(y,P,x,iw,nonzero,lam):
    siw = iw[nonzero]
    if sum(iw>1e-5)>sum(nonzero)+10:
        iw = iw*0.2
    elif abs(sum(iw>1e-5)-sum(nonzero))<2:
        iw = iw*0.01
    iw[nonzero] = siw
    P_iW = P*iw
    P_iW_P = dot(P,P_iW.T)
    x      = dot( P_iW.T , la.solve( lam*eye(y.size) + P_iW_P , y))
    iw     = x**2 + iw - sum( dot( P_iW.T , la.pinv(P_iW_P) ) * P_iW.T , axis=1)
    PC  = dot( P.T , la.pinv( lam*eye(y.size) + P_iW_P ))
    PCP = sum( PC * P.T , axis=1 )
    PCT = sum( PC * y , axis=1 )
    s = PCP - PCP**2/(PCP-1/(iw+1e-15))
    q = PCT - PCP*PCT/(PCP-1/(iw+1e-15))
    nonzero = q**2 > s
    return x,iw,nonzero

def s3IRLS_step(y,P,x,iw,lam):
    P_iW = P*iw
    P_iW_P = dot(P,P_iW.T)
    x      = dot( P_iW.T , la.solve( lam*eye(y.size) + P_iW_P , y))
    C = la.pinv( lam*eye(y.size) + P_iW_P)
    iw     = x**2 + iw - sum( dot( P_iW.T , C ) * P_iW.T , axis=1)
    PC  = dot( P.T , C )
    PCP = sum( PC * P.T , axis=1 )
    PCT = sum( PC * y , axis=1 )
    s = PCP - PCP**2/(PCP-1/(iw+1e-15))
    q = PCT - PCP*PCT/(PCP-1/(iw+1e-15))
    nonzero = q**2 > s
    siw = iw[nonzero]
    if sum(iw>1e-5)>sum(nonzero)+5:
        iw = iw*0.9
    elif sum(iw>1e-5)<sum(nonzero)-5:
        iw = iw*1.2
    else:
        iw = iw*0.01
    iw[nonzero] = siw
    return x,iw,nonzero