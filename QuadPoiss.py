"""
Linear-Quadratic-Linear-Exponential-Poisson model.

All functions return a SYMBOLIC representation (using theano) of 
some desired quantity, as a function of the symbolic arguments.

@author: kolia
"""
import theano.tensor  as Th
from theano.sandbox.linalg import matrix_inverse, det
from kolia_theano import eig


def quadratic_Poisson( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                       STA   = Th.dvector('STA')  , STC  = Th.dmatrix('STC'), 
                       N_spike = Th.dscalar('N_spike'), logprior = 0 , 
                       **other):
    '''
    The actual quadratic-Poisson model, as a function of theta and M, 
    with a barrier on the log-det term.
    '''
    ImM = Th.identity_like(M)-(M+M.T)/2
    ldet = Th.log( det( ImM) )
    return -0.5 * N_spike *( 
             ldet + logprior \
             - 1./(ldet+6)**2 \
             - Th.sum(Th.dot(matrix_inverse(ImM),theta) * theta) \
             + 2. * Th.sum( theta * STA ) \
             + Th.sum( M * (STC + Th.outer(STA,STA)) ))

def eig_barrier( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
                 STA   = Th.dvector('STA'), STC  = Th.dmatrix('STC'), **other):
     '''
     A barrier enforcing that the log-det of M should be > exp(-6), 
     and all the eigenvalues of M > 0.  Returns true if barrier is violated.
     '''
     ImM = Th.identity_like(M)-(M+M.T)/2
     w,v = eig( ImM )
     return 1-(Th.sum(Th.log(w))>-6)*(Th.min(w)>0)

def UV( U  = Th.dmatrix('U') , V1   = Th.dvector('V1') , V2 = Th.dvector('V2') , **result):
    '''
    Reparameterize theta and M as a function of U, V1 and V2.
    '''
    result['theta'] = Th.dot( U.T , V1 )
    result['M'    ] = Th.dot( V1 * U.T , (V2 * U.T).T )
    return result

def UVs(N):
    '''
    Reparameterize a list of N (theta,M) parameters as a function of a 
    common U,V2 and a matrix of N rows containing V1.
    '''
    def UV( U    = Th.dmatrix('U')   , V1  = Th.dmatrix('V1') , V2 = Th.dvector('V2') ,
            STAs = Th.dmatrix('STAs'), STCs = Th.dtensor3('STCs'), 
            N_spikes = Th.dvector('N_spikes'),  **other):
        return [{'theta':    Th.dot( U.T , V1[i,:] ) ,
                 'M'  :      Th.dot( V1[i,:] * U.T , (V2 * U.T).T ),
                 'STA':      STAs[i,:],
                 'STC':      STCs[i,:,:],
                 'N_spike':  N_spikes[i]/(Th.sum(N_spikes)) ,
                 'logprior': Th.sum(0.001*Th.log(V1)) } for i in range(N)]
    return UV

def lUVs(N):
    '''
    Reparameterize a list of N (theta,M) parameters as a function of a 
    common log(U),V2 and a matrix of N rows containing log(V1).
    '''
    def UV( lU   = Th.dmatrix('lU')  , lV1  = Th.dmatrix('lV1') , V2 = Th.dvector('V2') ,
            STAs = Th.dmatrix('STAs'), STCs = Th.dtensor3('STCs'), **other):
        U  = Th.exp(lU + 1e-10)
        V1 = Th.exp(lV1+ 1e-10)
        return [{'theta': Th.dot( U.T , V1[i] ) ,
                 'M'  :   Th.dot( V1[i] * U.T , (V2 * U.T).T ),
                 'STA':   STAs[i,:],
                 'STC':   STCs[i,:,:]} for i in range(N)]
    return UV

def eigs( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
          STA   = Th.dvector('STA')  , STC   = Th.dmatrix('STC'), **other):
    '''
    Return eigenvalues of I-sym(M), for display/debugging purposes.
    '''
    ImM = Th.identity_like(M)-(M+M.T)/2
    w,v = eig( ImM )
    return w

def ldet( theta = Th.dvector('theta'), M    = Th.dmatrix('M') ,
          STA   = Th.dvector('STA'), STC  = Th.dmatrix('STC'), **other):
    '''
    Return log-det of I-sym(M), for display/debugging purposes.
    '''
    ImM = Th.identity_like(M)-(M+M.T)/2
    w, v = eig(ImM)
    return Th.sum(Th.log(w))
