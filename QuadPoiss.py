"""
Linear-Quadratic-Linear-Exponential-Poisson model.

SYMBOLIC representations (using theano) of desired quantities

@author: kolia
"""
import theano.tensor  as Th
from theano.sandbox.linalg import matrix_inverse, det
from kolia_theano import eig, logdet

#from IPython.Debugger import Tracer; debug_here = Tracer()

def named( **variables ):
    if variables.has_key('other')  :  del variables['other']
    for name,v in variables.items():  v.name = name
    return variables

def LQLEP( theta   = Th.dvector()  , M    = Th.dmatrix() ,
           STA     = Th.dvector()  , STC  = Th.dmatrix(), 
           N_spike = Th.dscalar(), **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, without any barriers or priors.
    '''
    ImM = Th.identity_like(M)-(M+M.T)/2
    ldet = logdet(ImM)
    LQLEP = -0.5 * N_spike *( ldet \
             - Th.sum(Th.dot(matrix_inverse(ImM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) ))
    other.update(locals())
    return named( **other )

def LQLEP_wBarrier( LQLEP    = Th.dscalar(), ldet = Th.dscalar(), 
                    N_spike  = Th.dscalar(), ImM  = Th.dmatrix(),  **other):
    '''
    The actual Linear-Quadratic-Exponential-Poisson log-likelihood, 
    as a function of theta and M, 
    with a barrier on the log-det term and a prior.
    '''
    LQLEP_wPrior = LQLEP + 0.5 * N_spike * 1./(ldet+250.)**2.
    eigsImM,_ = eig( ImM )
    barrier   = 1-(Th.sum(Th.log(eigsImM))>-250)*(Th.min(eigsImM)>0)
    other.update(locals())
    return named( **other )

def eig_pos_barrier( barrier = Th.dscalar(), V1 = Th.dvector(), **other):
    '''
    A barrier enforcing that the log-det of M should be > exp(-6), 
    and all the eigenvalues of M > 0.  Returns true if barrier is violated.
    '''
    posV1_barrier = 1-(1-barrier)*(Th.min(V1.flatten())>0)
    other.update(locals())
    return named( **other )

def UV12( U    = Th.dmatrix(),  V1   = Th.dmatrix() , V2       = Th.dvector(),
          STAs = Th.dmatrix(),  STCs = Th.dtensor3(), N_spikes = Th.dvector(), **other):
    other.update(locals())
    return named( **other )

def UVs(N):
    '''
    Reparameterize a list of N (theta,M) parameters as a function of a 
    common U,V2 and a matrix of N rows containing V1.
    '''
    def UV( U    = Th.dmatrix(), V1   = Th.dmatrix() , V2 = Th.dvector() ,
            STAs = Th.dmatrix(), STCs = Th.dtensor3(), 
            N_spikes = Th.dvector(), **other):
       return [named( **{'theta':    Th.dot( U.T , V1[i,:] ) ,
                         'M'  :      Th.dot( V1[i,:] * U.T , (V2 * U.T).T ),
                         'STA':      STAs[i,:],
                         'STC':      STCs[i,:,:],
                         'N_spike':  N_spikes[i]/(Th.sum(N_spikes)) ,
                         'U' :       U } ) for i in range(N)]
    return UV

def linear_reparameterization( T  = Th.dtensor3() , u  = Th.dvector() , **other ):
    U = Th.sum( T*u , axis=2 )    # U = Th.tensordot(T,u,axes=0)
    other.update(locals())
    return named( **other )
