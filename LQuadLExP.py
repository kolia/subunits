"""
Linear-Quadratic-Linear-Exponential-Poisson model fitter.
@author: kolia
"""
from numpy  import add,reshape,concatenate,log,eye,isnan,ones_like
from numpy.linalg import inv, det, norm
from theano import function
import theano.tensor  as Th
import scipy.optimize as Opt


class posterior:
    def __init__(self,N,Nsub,NRGC,prior=1):
        self.N     = N
        self.Nsub  = Nsub
        self.NRGC  = NRGC
        U    = Th.dmatrix()                   # SYMBOLIC variables     #
        V1   = Th.dvector()                                            #
        V2   = Th.dvector()                                            #
        STA  = Th.dvector()                                            #
        STC  = Th.dmatrix()                                            #
        theta= Th.dot( U.T , V1 )                                      #
        M    = Th.dot( V1 * U.T , (V2 * U.T).T )                       #
        detM = Th.dscalar()
        invM = Th.dmatrix()
        invMtheta = Th.dot(theta,invM)
        
        post = (  Th.log(detM) \
                - 0.01 / (detM-0.1) \
                - Th.dot(invMtheta,theta) \
                + 2*Th.dot( theta.T , STA ) \
                + Th.sum( M.T * (STC + STA.T*STA) ))/2
        dpost_dM  = ( invM + invMtheta * invMtheta.T \
                    + 0.01 * detM * invM / ((detM-0.1)**2))/2

#        post      = Th.log(detM)
#        dpost_dM  = invM

#        post      = - Th.dot(invMtheta,theta)
#        dpost_dM  = invMtheta * invMtheta.T

#        post      = - 0.01 / (detM-0.3)
#        dpost_dM  =   0.01 * detM * invM / ((detM-0.3)**2)

#        post      = 2*Th.dot(theta.T,STA) + Th.sum(M.T*(STC + STA.T*STA))
#        dpost_dM  = 0



        def dpost(dX):
            return Th.grad( cost = post                   , wrt = dX , \
                            consider_constant=[invM,detM,STC,STA] ) \
                 - Th.grad( cost = Th.sum( dpost_dM * M ) , wrt = dX , 
                            consider_constant=[dpost_dM,STA,STC,invM,invMtheta])

        self.M          = function( [U,V2,V1]                  ,  M       ) #
        self.posterior  = function( [U,V2,V1,invM,detM,STA,STC],  post    ) #
        self.dpost_dU   = function( [U,V2,V1,invM,detM,STA,STC], dpost(U) ) #
        self.dpost_dV1  = function( [U,V2,V1,invM,detM,STA,STC], dpost(V1)) #
        self.dpost_dV2  = function( [U,V2,V1,invM,detM,STA,STC], dpost(V2)) #


    def callback(self,params,data):
        (U,V2,V1) = self.params(params,data[0])
        print 'posterior:' , self.f(params,data[0]) , \
              ' det Mk:', [ det(eye(self.N)-baye.M(U,V2,V1[i,:])) \
                                        for i in range(self.NRGC)]
        
    def U(self,params):
        return reshape( params , (self.Nsub,self.N) )

    def V1(self,params):
        return reshape( params , (self.NRGC,self.Nsub) )

    def params(self,params,data):
        return (self.U(params[0:self.N*self.Nsub]),
                params[self.N*self.Nsub:self.N*self.Nsub+self.Nsub],
                self.V1(params[self.N*self.Nsub+self.Nsub:]))

    def data(self,params,data): return data

#    def sum_RGC(self,op,g,(U,V2,V1),(N_spikes,STA,STC)):
#        return reduce( op ,  [ n * g(U,V2,V1[i,:], \
#                                     inv(eye(self.N)-self.M(U,V2,V1[i,:])), \
#                                     det(eye(self.N)-self.M(U,V2,V1[i,:])),sta,stc) \
#                                for i,(n,sta,stc) in \
#                                enumerate(zip(N_spikes,STA,STC))])

    def sum_RGC(self,op,g,(U,V2,V1),(N_spikes,STA,STC)):
        for i,(n,sta,stc) in enumerate(zip(N_spikes,STA,STC)):
            IM = eye(self.N)-self.M(U,V2,V1[i,:])
            term = n * g(U,V2,V1[i,:], inv(IM), det(IM), sta, stc)
#            if any(isnan(term.flatten())):
#                print 'oups'
#                term = None
##                raise ArithmeticError('nan')
            if i == 0:
                result = term
            else:                
                result = op( result , term )
        return result

    def     f (self, params, data):
        return -self.sum_RGC( add, self.posterior, 
                              self.params(params,data),
                              self.data  (params,data))

    def df_dU (self, params, data):
        return -self.sum_RGC( add, self.dpost_dU ,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV1(self, params, data):
        def concat(a,b): return concatenate((a,b))
        return -self.sum_RGC( concat, self.dpost_dV1,
                              self.params(params,data),
                              self.data  (params,data))

    def df_dV2(self, params, data):
        return -self.sum_RGC( add, self.dpost_dV2,
                              self.params(params,data),
                              self.data  (params,data))

    def df(self, params, data):
        return concatenate((self.df_dU(params,data).flatten(),
                            self.df_dV2(params,data),
                            self.df_dV1(params,data)))

    def MAP(self,params,data):
        def cb(para): return self.callback(para,data)
#        return Opt.fmin_ncg(self.f,params,fprime=self.df,avextol=1.1e-5,
#                            maxiter=100,args=data,
#                            callback=cb)
        return Opt.fmin_bfgs(self.f,params,fprime=self.df,gtol=1.1e-3,
                            maxiter=100,args=data,
                            callback=cb)

#    def Cbm1(self,U):
#        """Memoize Cbm1 = inv(Cb) for efficiency."""
#        if not all(self.memo_U == U):
#            self.memo_U = U
##            print self.Cb(U)
#            Cbm1 = inv(self.Cb(U))
#            self.memo_Cbm1 = 0.5 * (Cbm1 + Cbm1.T)
##            print amax(self.memo_Cbm1-self.memo_Cbm1.T)
#        return self.memo_Cbm1



class posterior_single(posterior):
    def data(self,params,data):
        X,Y,N_spikes,STA,STC = data
        return (N_spikes,STA,STC)


class posterior_dU(posterior_single):
    '''Optimization wrt U only'''
    def params(self,params,data):
        V2,V1,N_spikes,STA,STC = data
        return ( self.U(params), V2, V1)
        
    def df(self,params,data):
        return self.df_dU(params,data).flatten()

                    
class posterior_dV2(posterior_single):
    '''Optimization wrt V2 only'''
    def params(self,params,(U,V1,N_spikes,STA,STC)):
        return ( U, params, V1)

    def df(self,params,data):
        return self.df_dV2(params,data)
    


class posterior_dV1(posterior_single):
    '''Optimization wrt V1 only'''
    def params(self,params,(U,V2,N_spikes,STA,STC)):
        return ( U, V2, self.V1(params))

    def df(self,params,data):
        return self.df_dV1(params,data)




from simulate_data import LNLNP
from numpy import dot, ones , arange
import numpy.random   as R
import numpy.linalg   as L

V2 = 0.3
def NL(x): return x + V2 * x ** 2
(N_spikes,STA,STC), U, V1 = LNLNP(NL=NL,N=4)
Nsub, N    =  U.shape
NRGC, Nsub = V1.shape
V2 = V2 * ones((Nsub,))
                                        

#baye   = posterior_dU(N,Nsub,NRGC)
#data   = [ (V2, V1, N_spikes , STA , STC) ]
#index  = 0
#true_params = U.flatten()

#baye   = posterior_dV2(N,Nsub,NRGC)
#data   = [ ( U, V1, N_spikes , STA , STC ) ]
#index  = 1
#true_params = V2

baye   = posterior_dV1(N,Nsub,NRGC)
data   = [ ( U, V2, N_spikes , STA , STC) ]
index  = 2
true_params = V1.flatten()

#baye   = posterior(N,Nsub,NRGC)
#data   = [ (N_spikes , STA , STC) ]
#index  = slice(3)
#true_params = concatenate(( U.flatten() , V2 , V1.flatten() ))

# Check derivative
print
print 'Checking derivatives:'
df = baye.df(true_params,data[0])
UU = U.flatten()
dh = 0.00000001
ee = eye(len(true_params))*dh
for i in arange(len(true_params)):
    print (baye.f(true_params+ee[:,i],data[0])-baye.f(true_params,data[0]))/dh , df[i]


print
print 'U  : ', U
print
print 'V1 : ' , V1
print
print 'U.shape    = ', U.shape
print 'V2.shape   = ', V2.shape
print 'V1.shape   = ', V1.shape
print 'N_cones    = ', N
print 'N_subunits = ', Nsub
print 'N_RGC      = ', NRGC
print 'N_spikes   = ', N_spikes
print 'norm( Mk )  = '       , [ norm(eye(N)-baye.M(U,V2,V1[i,:])) \
                                        for i in range(NRGC)]

print
print 'true params :'
baye.callback(true_params,data)
params = true_params + 0.3 * R.randn(len(true_params))
print
print 'initial params :'
baye.callback(params,data)
print

opt_params = baye.MAP(params,data)
print 'log-likelihood of init params = ', baye.f(params,data[0])
print 'log-likelihood of true params = ', baye.f(true_params,data[0])

print 'init params:'
print  baye.params(params,data[0])[index]
print 'true params:'
print  baye.params(true_params,data[0])[index]
print 'opt  params:'
print  baye.params(opt_params ,data[0])[index]