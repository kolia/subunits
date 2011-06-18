"""
Linear-Quadratic-Linear-Exponential-Poisson model for Nuclear Norm optimizer
@author: kolia
"""

from numpy  import add, concatenate, eye, isnan, iscomplex,\
                   Inf, arange, max, min, minimum, log, size
#from numpy.linalg import inv, slogdet, det

import theano.tensor  as Th
from theano.sandbox.linalg import matrix_inverse
from kolia_theano import slogdet, eig

def quadratic_Poisson( theta = Th.dvector(), M    = Th.dmatrix() ,
                       STA   = Th.dvector(), STC  = Th.dmatrix()):

    IM = Th.identity_like(M)-M
    s, ldet = slogdet(IM)
    return -( ldet  \
             - 1./(ldet+6)**2 \
#             - Th.sum(Th.as_tensor_variable(Th.dot(matrix_inverse(IM),theta),ndim=2) * theta) \
             - Th.sum(Th.dot(matrix_inverse(IM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2.


def barrier( theta = Th.dvector(), M    = Th.dmatrix() ,
             STA   = Th.dvector(), STC  = Th.dmatrix() ):
     w,v = eig( Th.identity_like(M)-M )
     s,ldet = slogdet(Th.identity_like(M)-M)
     return 1-(ldet>-6)*(Th.min(w)>0)
#                 return (s+1)/2 * (ldet+100) < 0


def bar( theta = Th.dvector(), M    = Th.dmatrix() ,
             STA   = Th.dvector(), STC  = Th.dmatrix() ):
     w,v = eig( Th.identity_like(M)-M )
     s,ldet = slogdet(Th.identity_like(M)-M)
     return (ldet>-2,Th.min(w)>0,1-(ldet>-2)*(Th.min(w)>0))


def eigs( theta = Th.dvector(), M    = Th.dmatrix() ,
          STA   = Th.dvector(), STC  = Th.dmatrix()):
    w,v = eig( Th.identity_like(M)-M )
    return w


def ldet( theta = Th.dvector(), M    = Th.dmatrix() ,
          STA   = Th.dvector(), STC  = Th.dmatrix()):

    IM = Th.identity_like(M)-M
    s, ldet = slogdet(IM)
    return ldet


def data_match( theta = Th.dvector(), M    = Th.dmatrix() ,
                       STA   = Th.dvector(), STC  = Th.dmatrix()):

    return -(  2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2.


def quad_term( theta = Th.dvector(), M    = Th.dmatrix() ,
               STA   = Th.dvector(), STC  = Th.dmatrix()):
    IM = Th.identity_like(M)-M
    return Th.sum(Th.dot(matrix_inverse(IM),theta) * theta)


#def sum_RGC(self,op,g,(U,V2,V1),(N_spikes,STA,STC)):
#    result = None
#    for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC)):
#        IM = eye(self.N)-self.M(U,V2,V1[i,:])
##            detIM = det(IM)
##            print 'det(IM) : ', detIM
#        term = n * g(U,V2,V1[i,:], inv(IM), det(IM), sta, stc)
#        if any(isnan(term.flatten())):
#            print 'oups'
#            term = None
##                raise ArithmeticError('nan')
#        if result is not None and term is not None:
#            result = op( result , term )
#        else:
#            if term is not None:
#                result = term
#    if result is None:
#        return Inf
#    else:
#        return -result