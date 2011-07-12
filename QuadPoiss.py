"""
Linear-Quadratic-Linear-Exponential-Poisson model for Nuclear Norm optimizer
@author: kolia
"""

from numpy  import add, concatenate, eye, isnan, iscomplex, \
                   Inf, arange, max, min, minimum, log, size
#from numpy.linalg import inv, slogdet, det

import theano.tensor  as Th
from theano.sandbox.linalg import matrix_inverse, det
from kolia_theano import logdet, eig


def UV( U  = Th.dmatrix('U') , V1   = Th.dvector('V1') , V2 = Th.dvector('V2') , **result):
    result['theta'] = Th.dot( U.T , V1 )
    result['M'    ] = Th.dot( V1 * U.T , (V2 * U.T).T )
    return result


def UVs(N):
    def UV( U    = Th.dmatrix('U')   , V1  = Th.dmatrix('V1') , V2 = Th.dvector('V2') ,
            STAs = Th.dmatrix('STAs'), STCs = Th.dtensor3('STCs'), **other):
        return [{'theta': Th.dot( U.T , V1[i] ) ,
                 'M'  :   Th.dot( V1[i] * U.T , (V2 * U.T).T ),
                 'STA':   STAs[i,:],
                 'STC':   STCs[i,:,:]} for i in range(N)]
    return UV

def quadratic_Poisson( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                       STA   = Th.dvector('STA')  , STC  = Th.dmatrix('STC'), **other):

    ImM = Th.identity_like(M)-(M+M.T)/2
#    ldet = logdet( ImM)
    ldet = Th.log( det( ImM) )
    return -( ldet  \
             - 1./(ldet+6)**2 \
#             - Th.sum(Th.as_tensor_variable(Th.dot(matrix_inverse(ImM),theta),ndim=2) * theta) \
             - Th.sum(Th.dot(matrix_inverse(ImM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2.


def det_barrier( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                 STA   = Th.dvector('STA'), STC  = Th.dmatrix('STC'), **other):
     ImM = Th.identity_like(M)-(M+M.T)/2
     ldet = logdet( ImM)
     return (ldet+100) < 0


def eig_barrier( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                 STA   = Th.dvector('STA'), STC  = Th.dmatrix('STC'), **other):
     ImM = Th.identity_like(M)-(M+M.T)/2
     w,v = eig( ImM )
     return 1-(Th.sum(Th.log(w))>-6)*(Th.min(w)>0)


def eigs( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
#def eigs( M   , theta = Th.dvector('theta'),
          STA   = Th.dvector('STA')  , STC   = Th.dmatrix('STC'), **other):
    ImM = Th.identity_like(M)-(M+M.T)/2
    w,v = eig( ImM )
    return w


def ldet( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
          STA   = Th.dvector('STA'), STC  = Th.dmatrix('STC'), **other):
    ImM = Th.identity_like(M)-(M+M.T)/2
    w, v = eig(ImM)
    return Th.sum(Th.log(w))
