from numpy  import dot,sin,exp,std,eye,sum,add,all,newaxis,sqrt,ones, array
from numpy  import arange,transpose,fromfunction,reshape,concatenate
from numpy.linalg import inv
from theano import function
import theano.tensor  as Th
import numpy.random   as R
import scipy.linalg   as L
#import scipy          as S
import scipy.optimize as Opt


def simulate_LNLNP(
    N             = 10               ,  # number of cones, subunits & RGCs
    Sigma         = 1.               ,  # subunit & RGC connection width
    firing_rate   = 0.1              ,  # RGC firing rate
    T             = 100000           ): # number of time samples
    """Simulate spiking data for an LNLNP model.
    Then use this data to calculate the STAs and STCs.
    """
	
    spatial  = fromfunction( lambda x: exp(-0.5 * ((x-N/2)/Sigma)**2) , (N,) )
    U        = L.circulant( spatial )   # connections btw cones & subunits
    V        = U                        # connections btw subunits & RGCs

    U = U[0:10:2,:]
    V = V[0:10:2,0:10:2]
    NRGC = V.shape[0]

    U = U / sqrt(sum(U*U,axis=1))[:,newaxis]
    V = V / sqrt(sum(V*V,axis=1))[:,newaxis]

    X        = R.randn(N,T)                 # cone activations
#    b        = sin(dot(U,X))                # subunit activations
#    b        = exp(dot(U,X))                # subunit activations
#    b        = dot(U,X)                     # subunit activations
    b        = dot(U,X) + dot(U,X) ** 2     # subunit activations
    Y        = exp(dot(V,b))                # RGC activations
    Y        = Y * NRGC * T * firing_rate / sum(Y)

    print 'std( X ) = ', std(X)
    print 'std( U X ) = ', std(dot(U,X))
    print 'std( V b ) = ', std(dot(V,b))

    spikes   = R.poisson(Y)
    N_spikes = sum(spikes,1)
	
    STA = [ sum(X*spikes[i,:],1) / N_spikes[i] for i in arange(NRGC) ]

    STC = [ dot( X-STA[i][:,newaxis] , transpose((X-STA[i][:,newaxis])*Y[i,:])) \
                 / N_spikes[i] - eye(N)                for i in arange(NRGC) ]

    return ((N_spikes,STA,STC),U,V)


def sin_model(U, STA, STC):
    """b(s)  =  sin( U s )"""
    STAB = Th.exp(-0.5*Th.sum(Th.dot(U,STC)*U,axis=1)) * Th.sin(Th.dot(U,STA))
    bbar = Th.zeros_like(STAB)
    eU   = Th.exp( -0.5 * Th.sum( U * U , axis=1 ) )
    Cb   = 0.5 * (Th.sinh(Th.dot(U,U.T))*eU).T*eU
    regular  = 0.0000000001*Th.sum( Th.cosh(Th.sum(U*U,axis=1)) )
    return (STAB,bbar,Cb,regular)


def exp_model(U, STA, STC):
    """b(s)  =  exp( U s )"""
    bbar = Th.exp(0.5* Th.sum( U * U , axis=1 ))
    STAB = Th.exp(0.5* Th.sum(Th.dot(U,STC)*U,axis=1) + Th.dot(U,STA))
    Cb   = (Th.exp(0.5* Th.dot(U,U.T))*bbar).T*bbar
    regular  = 0.0000000001*Th.sum( Th.cosh(Th.sum(U*U,axis=1)) )
    return (STAB,bbar,Cb,regular)


def lin_model(U, STA, STC):
    """b(s)  =  U s"""
    STAB = Th.dot( U , STA )
    bbar = Th.zeros_like(STAB)
    Cb   = Th.dot(U,U.T)
    regular  = 0 #0.0000000001*Th.sum(U*U)
    return (STAB,bbar,Cb,regular)


class posterior_b:
    def __init__(self,model,prior=1):
        self.memo_U    = None
        self.memo_Cbm1 = None
        self.prior     = prior
        
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        (STAB,bbar,Cb,regular) = model(U, STA, STC)                     #
        Cbm1       = Th.dmatrix()                                       #
        STABC      = Th.dot(Cbm1,(STAB-bbar))                           #
        posterior  = Th.sum(STABC*(STAB-bbar)) - regular                #
        dposterior = 2*posterior - Th.sum( Th.outer(STABC,STABC) * Cb ) - regular #
        dposterior = Th.grad( cost              = dposterior,           #
                              wrt               = U         ,           #
                              consider_constant = [STABC]   )           #

        self.Cb         = function( [U]             ,  Cb       )
        # log-posterior term  (STAB - bbar).T inv(Cb) (STAB - bbar)
        self.posterior  = function( [U,STA,STC,Cbm1],  posterior)
        # gradient of log-posterior term
        self.dposterior = function( [U,STA,STC,Cbm1], dposterior)

    def Cbm1(self,U):
        """Memoize Cbm1 = inv(Cb) for efficiency."""
        if not all(self.memo_U == U):
            self.memo_U = U
#            print self.Cb(U)
            Cbm1 = inv(self.Cb(U))
            self.memo_Cbm1 = 0.5 * (Cbm1 + Cbm1.T)
#            print amax(self.memo_Cbm1-self.memo_Cbm1.T)
        return self.memo_Cbm1

    def sum_RGCs(self, f, U, (N_spikes,STA,STC)):
        U = reshape(U,(-1,len(STA[0])))
        Cbm1 = self.Cbm1(U)
        return reduce( add ,  [ n**2/(n+self.prior) * f(U,sta,stc,Cbm1).flatten() \
                               for n,sta,stc in zip(N_spikes,STA,STC)])

    def callback(self,U):
        print sum( U*U, axis=1)

    def  f(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.posterior , U, (N_spikes,STA,STC))
    def df(self, U, N_spikes,STA,STC): return -self.sum_RGCs( self.dposterior, U, (N_spikes,STA,STC))
    def MAP(self,data,x0): return Opt.fmin_ncg(self.f,x0.flatten(),fprime=self.df,avextol=1.1e-3,args=data,callback=self.callback)


#data, U, V = simulate_LNLNP()
#baye    = posterior_b(lin_model,prior=1.)
##baye    = posterior_b(exp_model,prior=1.)
#
#N_spikes,STA,STC=data
#print baye.f(U,N_spikes,STA,STC)
#
### Check derivative
##df = baye.df(U,N_spikes,STA,STC)
##UU = U.flatten()
##dh = 0.00000001
##ee = eye(100)*dh
##for i in arange(100):
##    print (baye.f(UU+ee[:,i],N_spikes,STA,STC)-baye.f(UU,N_spikes,STA,STC))/dh , df[i]
#
#x0  = U + R.randn(5,10)
##x0  = array( STA ).T
#
#x  = baye.MAP(data,x0.flatten())
#x  = reshape( x , (-1,10) )


class posterior_quad:
    '''Joint optimization in U, V2 and V1'''
    def _init_(self,N,Nsub,NRGC,prior=1):
        self.N     = N
        self.Nsub  = Nsub
        self.NRGC  = NRGC
        U   = Th.dmatrix()                   # SYMBOLIC variables       #
        V1  = Th.dvector()                                              #
        V2  = Th.dvector()                                              #
        STA = Th.dvector()                                              #
        STC = Th.dmatrix()                                              #
        theta = Th.dot( U.T , V1 )                                      #
        UV1U  = Th.dot( U , theta )                                     #
        posterior  = -0.5 * Th.sum( V1*V2 * U*U ) \
                     -0.5 * Th.sum( UV1U * UV1U * V1*V2 ) \
                     + Th.dot( theta.T , STA ) \
                     + Th.sum( Th.dot( U.T , V1*V2*U ) * (STC + STA.T*STA) )
        dpost_dU  = Th.grad( cost           = posterior ,               #
                             wrt            = U         )               #
        dpost_dV1 = Th.grad( cost           = posterior ,               #
                             wrt            = V1        )               #
        dpost_dV2 = Th.grad( cost           = posterior ,               #
                             wrt            = V2        )               #
        self.posterior  = function( [U,V2,V1,STA,STC],  posterior)      #
        self.dpost_dU   = function( [U,V2,V1,STA,STC], dpost_dU  )      #
        self.dpost_dV1  = function( [U,V2,V1,STA,STC], dpost_dV1 )      #
        self.dpost_dV2  = function( [U,V2,V1,STA,STC], dpost_dV2 )      #


    def callback(self,params):
        print sqrt( sum( params ** 2 ) )
        
    def U(self,params):
        return reshape( params , (self.Nsub,self.N) )

    def V1(self,params):
        return reshape( params , (self.NRGC,self.Nsub) )

    def params(self,params,data):
        return (self.U(params[0:self.N*self.Nsub]),
                params[self.N*self.Nsub:self.N*self.Nsub+self.Nsub],
                self.V1(params[self.N*self.Nsub+self.Nsub:]))

    def data(self,params,data): return data

    def sum_RGC(self,op,f,(U,V2,V1),(N_spikes,STA,STC)):
        return reduce( op ,  [ n * f(U,V2,V1[i,:],sta,stc) \
                                for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC))])

    def     f (self, params, data):
        return -self.sum_RGC( add, self.posterior, self.params(params,data), self.data(params,data))

    def df_dU (self, params, data):
        return -self.sum_RGC( add, self.dpost_dU , self.params(params,data), self.data(params,data))

    def df_dV1(self, params, data):
        def concat(a,b): return concatenate((a,b))
        return -self.sum_RGC( concat, self.dpost_dV1, self.params(params,data), self.data(params,data))

    def df_dV2(self, params, data):
        return -self.sum_RGC( add, self.dpost_dV2, self.params(params,data), self.data(params,data))

    def df(self, params, data):
        return concatenate((self.df_dU(params,data),
                            self.df_dV2(params,data),
                            self.df_dV1(params,data)))

    def MAP(self,data,x0):
        return Opt.fmin_ncg(self.f,x0,fprime=self.df,avextol=1.1e-3,
                            args=data,callback=self.callback)

class posterior_quad_single(posterior_quad):
    def data(self,params,(X,Y,N_spikes,STA,STC)):
        return (N_spikes,STA,STC)    

class posterior_quad_dU(posterior_quad_single):
    '''Optimization wrt U only'''
    def params(self,params,(V2,V1,N_spikes,STA,STC)):
        return ( self.U(params), V2, V1)
                    
class posterior_quad_dV2(posterior_quad_single):
    '''Optimization wrt V2 only'''
    def params(self,params,(U,V1,N_spikes,STA,STC)):
        return ( U, params, V1)

class posterior_quad_dV1(posterior_quad_single):
    '''Optimization wrt V1 only'''
    def params(self,params,(U,V2,N_spikes,STA,STC)):
        return ( U, V2, self.V1(params))


data, U, V = simulate_LNLNP()
Nsub, N    = U.shape
NRGC, Nsub = V.shape
baye       = posterior_quad(N,Nsub,NRGC)

V2 = ones((Nsub,))

N_spikes,STA,STC=data
print baye.f(U,V2,V.T,N_spikes,STA,STC)

## Check derivative
#df = baye.df(U,N_spikes,STA,STC)
#UU = U.flatten()
#dh = 0.00000001
#ee = eye(100)*dh
#for i in arange(100):
#    print (baye.f(UU+ee[:,i],N_spikes,STA,STC)-baye.f(UU,N_spikes,STA,STC))/dh , df[i]

x0  = U + R.randn(5,10)
#x0  = array( STA ).T

x  = baye.MAP(data,x0.flatten())
x  = reshape( x , (-1,10) )
        