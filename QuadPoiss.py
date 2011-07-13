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

def exponentiate(lU  = Th.dmatrix('lU') , lV1  = Th.dmatrix('lV1'), **result ):
    result['U' ] = Th.exp(lU)
    result['V1'] = Th.exp(lV1)
    return result

def UVs(N):
    def UV( U    = Th.dmatrix('U')   , V1  = Th.dmatrix('V1') , V2 = Th.dvector('V2') ,
            STAs = Th.dmatrix('STAs'), STCs = Th.dtensor3('STCs'), **other):
        return [{'theta': Th.dot( U.T , V1[i] ) ,
                 'M'  :   Th.dot( V1[i] * U.T , (V2 * U.T).T ),
                 'STA':   STAs[i,:],
                 'STC':   STCs[i,:,:]} for i in range(N)]
    return UV

def lUVs(N):
    def UV( lU   = Th.dmatrix('lU')  , lV1  = Th.dmatrix('lV1') , V2 = Th.dvector('V2') ,
            STAs = Th.dmatrix('STAs'), STCs = Th.dtensor3('STCs'), **other):
        U  = Th.exp(lU )
        V1 = Th.exp(lV1)
        return [{'theta': Th.dot( U.T , V1[i] ) ,
                 'M'  :   Th.dot( V1[i] * U.T , (V2 * U.T).T ),
                 'STA':   STAs[i,:],
                 'STC':   STCs[i,:,:]} for i in range(N)]
    return UV

def positive_barrier( U = Th.dmatrix('U' ), V1  = Th.dmatrix('V1'),
                     V2 = Th.dvector('V2'), **other):
     return 1-Th.prod(Th.concatenate([U.flatten()>0,V1.flatten()>0,V2.flatten()>0]))

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

def positive( U = Th.dmatrix('U' ), V1 = Th.dmatrix('V1'),
             V2 = Th.dvector('V2'), **other):
    return Th.sum( 0.00000001/Th.concatenate([U.flatten(),V1.flatten(),V2.flatten()])**0.01 )
#    return Th.sum( 0.000000001*Th.log(Th.concatenate([U.flatten(),V1.flatten(),V2.flatten()]) ) )

def positive_quadratic_Poisson( U = Th.dmatrix('U' ),   V1    = Th.dmatrix('V1'),
                               V2 = Th.dvector('V2'), 
                            theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                            STA   = Th.dvector('STA'  ), STC  = Th.dmatrix('STC'), **other):

    ImM = Th.identity_like(M)-(M+M.T)/2
#    ldet = logdet( ImM)
    ldet = Th.log( det( ImM) )
    return -( ldet - positive(U=U,V1=V1,V2=V2) \
             - 1./(ldet+6)**2 \
#             - Th.sum(Th.as_tensor_variable(Th.dot(matrix_inverse(ImM),theta),ndim=2) * theta) \
             - Th.sum(Th.dot(matrix_inverse(ImM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) )) / 2.



def eig_positive_barrier( U = Th.dmatrix('U' ), V1  = Th.dmatrix('V1'),
                         V2 = Th.dvector('V2'), 
                      theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                      STA   = Th.dvector('STA'  ), STC  = Th.dmatrix('STC'), **other):
     ImM = Th.identity_like(M)-(M+M.T)/2
     w,v = eig( ImM )
     eigbar = 1-(Th.sum(Th.log(w))>-6)*(Th.min(w)>0)
     posbar = 1-Th.prod(Th.concatenate([U.flatten()>0,V1.flatten()>0,V2.flatten()>0]))
     return eigbar + posbar

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
