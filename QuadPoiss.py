"""
Linear-Quadratic-Linear-Exponential-Poisson model.

SYMBOLIC representations (using theano) of desired quantities

@author: kolia
"""

import theano.tensor  as Th
from theano.sandbox.linalg import matrix_inverse, det
from kolia_theano import eig, logdet

# from IPython.Debugger import Tracer; debug_here = Tracer()

def named( **variables ):
    if variables.has_key('other')  :  del variables['other']
    for name,v in variables.items():  v.name = name
    return variables

def LQLEP_input(**other):
    theta   = Th.dvector()
    M       = Th.dmatrix()
    STA     = Th.dvector()
    STC     = Th.dmatrix()
    N_spike = Th.dscalar()
    Cm1     = Th.dmatrix()
    other.update(locals())
    return named( **other )

def LNP(   theta = Th.dvector(),  STA = Th.dvector(), 
         N_spike = Th.dscalar(),  C   = Th.dmatrix(),  **other):
    '''
    LNP log-likelihood, as a function of theta.  Minimizer is the STA.
    '''
    LNP = N_spike *( 0.5* Th.sum(Th.dot(C,theta) * theta) - Th.sum( theta * STA ))
    other.update(locals())
    return named( **other )

def LQLEP( theta   = Th.dvector()  , M    = Th.dmatrix() ,
           STA     = Th.dvector()  , STC  = Th.dmatrix() , 
           N_spike = Th.dscalar()  , Cm1  = Th.dmatrix() , **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, without any barriers or priors.
    '''
#    ImM = Th.identity_like(M)-(M+M.T)/2
    ImM = Cm1-(M+M.T)/2
    ldet = logdet(ImM)
    LQLEP = -0.5 * N_spike *( ldet - logdet(Cm1) \
             - Th.sum(Th.dot(matrix_inverse(ImM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) ))
    other.update(locals())
    return named( **other )

def LQLEP_wBarrier( LQLEP    = Th.dscalar(), ldet = Th.dscalar(), v1 = Th.dvector(), 
                    N_spike  = Th.dscalar(), ImM  = Th.dmatrix(),  U = Th.dmatrix(),
                    V2       = Th.dvector(),    u = Th.dvector(),  C = Th.dmatrix(),
                    **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, 
    with a barrier on the log-det term and a prior.
    '''
    sq_nonlinearity = V2**2.*Th.sum( Th.dot(U,C)*U, axis=[1])  #Th.sum(U**2,axis=[1])
    nonlinearity = V2 * Th.sqrt( Th.sum( Th.dot(U,C)*U, axis=[1])) #Th.sum(U**2,axis=[1]) )
    if other.has_key('ub'):
        LQLEP_wPrior = LQLEP + 0.5 * N_spike * ( 1./(ldet+250.)**2. \
                     - 0.000001 * Th.sum(Th.log(1.-4*sq_nonlinearity))) \
                     + 1. * Th.sum( (u[2:]+u[:-2]-2*u[1:-1])**2. ) \
                     + 1. * Th.sum( (other['ub'][2:]+other['ub'][:-2]-2*other['ub'][1:-1])**2. )
    else:
        LQLEP_wPrior = LQLEP + 0.5 * N_spike * ( 1./(ldet+250.)**2. \
                     - 0.000001 * Th.sum(Th.log(1.-4*sq_nonlinearity))) #\
#                     + 1. * Th.sum( (u[2:]+u[:-2]-2*u[1:-1])**2. )
    #                 + 0.0001*Th.sum( V2**2 )
    eigsImM,barrier = eig( ImM )
    barrier   = 1-(Th.sum(Th.log(eigsImM))>-250) * \
                  (Th.min(eigsImM)>0) * (Th.max(4*sq_nonlinearity)<1)
    other.update(locals())
    return named( **other )

def LQLEP_positiveV1( LQLEP_wPrior = Th.dscalar(), barrier = Th.dscalar(),
                      v1 = Th.dvector(), **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, 
    with a barrier on the log-det term and a prior.
    '''
    LQLEP_positiveV1   = LQLEP_wPrior - 0.0001 * Th.sum(Th.log(v1))
    barrier_positiveV1 = 1-((1 - barrier) * (Th.min(v1.flatten())>=0))
    other.update(locals())
    return named( **other )

def LQLEP_positive_u( LQLEP_wPrior = Th.dscalar(), barrier = Th.dscalar(),
                     u = Th.dvector(), **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, 
    with a barrier on the log-det term and a prior.
    '''
    if other.has_key('ub'):
        LQLEP_positive_u   = LQLEP_wPrior - 0.0001 * Th.sum(Th.log(u)) \
                                          - 0.0001 * Th.sum(Th.log(other['ub']))
        barrier_positive_u = 1-((1 - barrier) * (Th.min(u.flatten())>=0) \
                                              * (Th.min(other['ub'].flatten())>=0))
    else:
        LQLEP_positive_u   = LQLEP_wPrior - 0.0001 * Th.sum(Th.log(u))
        barrier_positive_u = 1-((1 - barrier) * (Th.min(u.flatten())>=0))
    other.update(locals())
    return named( **other )

def eig_pos_barrier( barrier = Th.dscalar(), V1 = Th.dvector(), **other):
    '''
    A barrier enforcing that the log-det of M should be > exp(-6), 
    and all the eigenvalues of M > 0.  Returns true if barrier is violated.
    '''
    posV1_barrier = 1-(1-barrier)*(Th.min(V1.flatten())>=0)
    other.update(locals())
    return named( **other )

def UV12( U    = Th.dmatrix(),  V1   = Th.dmatrix() , V2       = Th.dvector(),
          STAs = Th.dmatrix(),  STCs = Th.dtensor3(), N_spikes = Th.dvector(), **other):
    other.update(locals())
    return named( **other )

def thetaM( U   = Th.dmatrix(), v1  = Th.dvector() , V2 = Th.dvector() ,
            STA = Th.dvector(), STC = Th.dmatrix(),  N_spike = Th.dscalar(), **other):
    theta = Th.dot( U.T , v1 )
    M     = Th.dot( v1 * U.T , (V2 * U.T).T )
    other.update(locals())
    return named( **other )


def UV12_input(V1   = Th.dmatrix() , STAs = Th.dmatrix(), 
               STCs = Th.dtensor3(), N_spikes = Th.dvector(), **other):
    other.update(locals())
    return named( **other )

def UVi(i , V1   = Th.dmatrix() , STAs = Th.dmatrix(), STCs = Th.dtensor3(), 
        N_spikes = Th.dvector(), **other):
    '''
    Reparameterize a list of N (theta,M) parameters as a function of a 
    common U,V2 and a matrix of N rows containing V1.
    '''
    return named( **{'v1'  :    V1[i,:] ,
                     'STA' :    STAs[i,:],
                     'STC' :    STCs[i,:,:],
                     'N_spike': N_spikes[i]/(Th.sum(N_spikes))} )

def linear_reparameterization( T  = Th.dtensor3() , u  = Th.dvector() , 
                                     **other ):
#                                b = Th.dvector() ,  ub = Th.dvector(), **other ): 
#    U = ( Th.sum( T*ub  , axis=2 ).T * b  ).T + Th.sum( T*u , axis=2 )
    U = Th.sum( T*u , axis=2 )    # U = Th.tensordot(T,u,axes=0)
    other.update(locals())
    return named( **other )

def u2V2_parameterization( T  = Th.dtensor3() , V2 = Th.dvector() ,
                           u  = Th.dvector() ,  ub = Th.dvector() ,
#                                   b = Th.dvector()
                           **other ):
#    Ub = Th.sum( T*ub , axis=2 )
#    Uc = Th.sum( T*uc , axis=2 )
    U  = Th.sum( T*u  , axis=2 ) + ( Th.sum( T*ub  , axis=2 ).T * V2  ).T
#    U  = ( Th.sum( T*ub  , axis=2 ).T * b  ).T + Th.sum( T*u  , axis=2 )
#       + ( Th.sum( T*uc , axis=2 ).T * V2 ).T
    other.update(locals())
    return named( **other )

def u2c_parameterization( T = Th.dtensor3() , V2 = Th.dvector() ,
                          u = Th.dvector()  , uc = Th.dvector() ,
                          c = Th.dvector()  , **other ):
#    Ub = Th.sum( T*ub , axis=2 )
#    Uc = Th.sum( T*uc , axis=2 )
    U  = Th.sum( T*u  , axis=2 ) + ( Th.sum( T*uc  , axis=2 ).T * c  ).T
#    U  = ( Th.sum( T*ub  , axis=2 ).T * b  ).T + Th.sum( T*u  , axis=2 )
#       + ( Th.sum( T*uc , axis=2 ).T * V2 ).T
    other.update(locals())
    return named( **other )

def u2cd_parameterization( T = Th.dtensor3() , V2 = Th.dvector() ,
                           u = Th.dvector()  , uc = Th.dvector() ,
                           c = Th.dvector()  , ud = Th.dvector() ,
                           d = Th.dvector()  , **other ):
#    Ub = Th.sum( T*ub , axis=2 )
#    Uc = Th.sum( T*uc , axis=2 )
    U  = Th.sum( T*u  , axis=2 ) + ( Th.sum( T*uc  , axis=2 ).T * c  ).T \
                                 + ( Th.sum( T*ud  , axis=2 ).T * d  ).T
#    U  = ( Th.sum( T*ub  , axis=2 ).T * b  ).T + Th.sum( T*u  , axis=2 )
#       + ( Th.sum( T*uc , axis=2 ).T * V2 ).T
    other.update(locals())
    return named( **other )
