import kolia_theano
reload(kolia_theano)

import simulate_data
reload(simulate_data)

import optimize
reload(optimize)

import numpy as np
from numpy.linalg import svd, inv
from scipy.linalg import orth
import numpy.random as R
import pylab as p

from matplotlib.ticker import *

#from IPython.Debugger import Tracer; debug_here = Tracer()

import QuadPoiss
reload(QuadPoiss)
from   QuadPoiss import quadratic_Poisson, eig_barrier

import FALMS
reload(FALMS)

############################
# Setting up simulated data

# Generate stimulus , spikes , STA , STC
V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )
(N_spikes,STA,STC), U, V1, bbar, Cb , STAB = simulate_data.LNLNP(NL=NL,N=24)
Nsub, Ncone =  U.shape
NRGC, Nsub  =  V1.shape

# True parameters
true   = [{ 'theta' : np.dot( U.T , V1[i,:] ) , \
            'M' : 0.1*np.dot( U.T * V1[i,:] , U ) } for i in range(NRGC)]

# SVD of stacked STAs and STCs
A = np.concatenate([N*np.reshape(sta,(1,len(sta))) for N,sta in zip(N_spikes,STA)])
B = np.concatenate([N*stc for N,sta,stc in zip(N_spikes,STA,STC)])
C = np.concatenate((A,B))
YU,S,VI = svd(C)


######################
# Function definitions

def callback( objective , params ):
    '''Print information once per iteration.'''
    print 'Objective: ' , objective.f(params) , '  barrier: ', objective.barrier(params)



def MaxLike(init_params,data):
    '''Maximum Likelihood optimization of parameters theta and M'''
    print
    print
    print 'Starting Max-Likelihood optimization in theta and M, compiling...'
    # Compile symbolic objectives into numerical objective.
    objective = kolia_theano.Objective(init_params=init_params[0],differentiate=['f'],
                                       callback=callback,
                                       f=quadratic_Poisson, barrier=eig_barrier)
    # Set fixed arguments STA and STC in optimizer object.
    optimizers = [ optimize.optimizer( objective.where(**dat) ) for dat in data ]
    
    # Maximize likelihood separately for each RGC; run the optimization twice just to be sure.
    params = init_params
    for i in range(2):
        params = [opt(init_params=par) for opt,par in zip(optimizers,params)]
        
    # Unpack flat parameters into meaningful dictionary.
    params = [objective.unflat(par) for par in params]
    
    # print norm of inferred and true theta
    optU = [param['theta'] for param in params]
    print
    print 'true    ||theta||^2  : ', np.sum(U*U,axis=1)
    print 'optimal ||theta||^2  : ', [np.sum(optu*optu) for optu in optU]
    print
    print
    return params


def analytical_ML(data):
    '''Analytical expression of Maximum Likelihood'''
    return [{ 'theta': np.dot( inv(dat['STC']) , dat['STA'] ) ,
              'M': np.eye(Ncone) - inv(dat['STC']) } for dat in data]


def MaxLike_L1( init_params, data , rho=1., mu=1e-5, maxiter=200, ftol=1e-7):
    '''L1-regularized Max-Likelihood optimization of parameters theta and M'''
    print
    print
    print 'Starting L1-regularized Max-Likelihood opt in theta and M, compiling...'
    # Compile symbolic objectives into numerical objective.
    objective = kolia_theano.Objective(init_params=init_params,differentiate=['f'],
                                       callback=callback,
                                       f=quadratic_Poisson, barrier=eig_barrier) 
    obj = objective.where(**data)
    optimize_L2 = FALMS.add_L2_term( obj )
    reg_L1 = FALMS.L1_L2( rho )
    
    current_X  = obj.flat(init_params)
    old_X      = current_X
    unskipped  = current_X
    objval_old = 1e10
    falmer = FALMS.initialize( current_X )
    for j in range(maxiter):
        oldfalmer   = falmer
        falmer = FALMS.step( optimize_L2 , reg_L1 , mu , oldfalmer )
        objval = obj.f(falmer[0])+reg_L1.f(falmer[0])
        L2change= np.sum((falmer[0]-old_X)**2) + np.sum((falmer[0]-current_X)**2)
        print 'FALMS STEP ' , j, 'with rho=',rho,' mu=',mu,'  OBJECTIVE: ', \
        objval,' (old val: ',objval_old,')'
        print 'L2 delta-params: ' , L2change,
        if falmer[6]:
            print '  skipped'
        else:
            unskipped = falmer[0]
            print '   L0: ', falmer[0][falmer[0]<>0].shape
        print
        if mu<1e-8 or (j>30 and L2change < ftol): break
        old_X = current_X
        current_X = falmer[0]
        if objval>objval_old:
            mu = mu/2
        if objval>objval_old+0.2:
            falmer = oldfalmer
        else:
            objval_old  = objval
    result = obj.unflat( unskipped )
    print result
    print
    print
    return result


def plot_vectors(params,title='Inferred thetas',figure=3):
    '''Takes a list of lists of vectors, plots one vector per subplot.'''
    p.figure(figure)
    Nsub  = len(params[0])
    Ncone = params[0][0].shape[0]
    for i in range(Nsub):
        vlist = sum( [[np.arange(Ncone),q[i]] for q in params] , [])
        ax = p.subplot(Nsub,1,i+1)
        p.plot(*vlist)
        ax.yaxis.set_major_locator( MaxNLocator(nbins=2) )
        if i == 0:  p.title(title)
        p.show()
    p.xlabel('Cone space')


def plot_matrices(mlist,title='Plot of matrix',figure=4):
    '''Takes a list of matrices, plots one matrix line per subplot.'''
    p.figure(figure)
    Nsub,Ncone  = mlist[0].shape
    for i in range(Nsub):
        vlist = sum( [[np.arange(Ncone),m[i,:]] for m in mlist] , [] )
        ax = p.subplot(Nsub,1,i+1)
        p.plot(*vlist)
        ax.yaxis.set_major_locator( MaxNLocator(nbins=2) )
        if i == 0:  p.title(title)
        p.show()
    p.xlabel('Cone space')

def subspace(A,B):
    '''Angle between subspaces, defined by matrix columns.'''
    u,s,v = svd(np.dot(np.transpose(orth(A)), orth(B)))
    return np.arccos(min(s.min(), 1.0))

def projection(uu,X):
    '''Orthogonal projection of vectors onto subspace.'''
    X = orth(X)
    return np.dot(X,np.dot(uu,X))

def plot_projection_of_U( onto = np.transpose(VI[0:Nsub,:]), name ='svd(STA,STC)', figure=5):
    p.figure(figure)
    nnn = U.shape[1]
    for i in range(Nsub):
        p.subplot(Nsub,1,i+1)
        p.plot(np.arange(nnn),U[i,:].flatten(),'b',np.arange(nnn), 
               projection(np.transpose(U[i,:]),onto).flatten(),'rs')
        if i==0: p.title('Subunit RFs and projection onto '+name)
        p.show()
    p.xlabel('Cone space')


######################
# Actual calculations


data  = [{ 'STA':STA[i] , \
           'STC':STC[i] } for i in range(NRGC)]
init_params = [{'theta':data[i]['STA'] * 0.1 , \
                'M':data[i]['STC'] * 0.1} for i in range(NRGC)]

params = init_params[0]
paramL1 = {}
#for rho in 1e-7*np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.2,1.5,2.,3.,5.]):
for rho in np.array([0.05,0.1,0.2,0.5,1.,2.,5.,10.]):
#for rho in np.array([0.1,0.5,1.,1.5,2.,2.5,3.]):
    params = MaxLike_L1(params,data[0],rho=rho,mu=1e-4,maxiter=200)
    paramL1[rho] = params
    print [ (rh,par['M'][par['M']<>0].shape) for rh,par in sorted(paramL1.items())]


#params = MaxLike(init_params,data)
#
#analytical = analytical_ML(data)
#
#
#
#Nproj = Nsub
#T     = VI[0:Nproj,:]
#projected_data  = [{ 'STA':np.dot(T,STA[i]) , \
#                     'STC':np.dot(np.dot(T,STC[i]),np.transpose(T)) } for i in range(NRGC)]
#projected_init_params = [{'theta':projected_data[i]['STA'] * 0.1 , \
#                          'M':    projected_data[i]['STC'] * 0.1} for i in range(NRGC)]
#
#projected_params = MaxLike(projected_init_params,projected_data)
#
#reinjected_params = [{'theta': np.dot(q['theta'],T)  ,  \
#                      'M':np.dot(np.dot(np.transpose(T),q['M']),T)} for q in projected_params]
#
#reinjected_A = np.concatenate([np.concatenate([np.array([q['theta']]),q['M']]) \
#                                for q in reinjected_params])
#
#reinjected_UA,reinjected_SA,reinjected_VA = svd(reinjected_A)
#
#plot_vectors([[q['theta'] for q in true      ], [q['theta'] for q in params           ], \
#              [q['theta'] for q in analytical], [q['theta'] for q in reinjected_params]])
#
#plot_matrices([true[0]['M'], params[0]['M'], analytical[0]['M'], reinjected_params[0]['M']])
#
#plot_projection_of_U( onto = np.transpose(reinjected_VA[0:Nsub,:]), name='svd(theta,M)', figure=5)
#plot_projection_of_U( onto = np.transpose(           VI[0:Nsub,:]), name='svd(STA,STC)', figure=6)