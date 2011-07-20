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
import cPickle

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

# quadratic Poisson LL in theta,M:
objective = kolia_theano.Objective(init_params=init_params[0],differentiate=['f'],
                                   f=quadratic_Poisson, barrier=eig_barrier) 

# fix particular data:
obj = objective.where(**data[0])

# LL + quadratic optimizer, for FALMS
optimize_L2 = FALMS.add_L2_term( obj )

# callback after each FALMS iteration:
def falms_callback(falmer):  print '   L0: ', falmer[1][falmer[1]<>0].shape[0]

r = {'L1':[{} for _ in init_params] , 'true':true}

def save(obj):
    savefile = open('../../../Desktop/results.pyckle','w')
    cPickle.dump(r,savefile)  
    savefile.close()


r['ML'] = MaxLike(init_params,data)

r['analytical'] = analytical_ML(data)



Nproj = Nsub
T     = VI[0:Nproj,:]
projected_data  = [{ 'STA':np.dot(T,STA[i]) , \
                     'STC':np.dot(np.dot(T,STC[i]),np.transpose(T)) } for i in range(NRGC)]
projected_init_params = [{'theta':projected_data[i]['STA'] * 0.1 , \
                          'M':    projected_data[i]['STC'] * 0.1} for i in range(NRGC)]

projected_params = MaxLike(projected_init_params,projected_data)

r['projected_ML'] = [{'theta': np.dot(q['theta'],T)  ,  \
                      'M':np.dot(np.dot(np.transpose(T),q['M']),T)} for q in projected_params]

reinjected_A = np.concatenate([np.concatenate([np.array([q['theta']]),q['M']]) \
                                for q in r['projected_ML']])

reinjected_UA,reinjected_SA,reinjected_VA = svd(reinjected_A)


for i,params in enumerate(init_params):
    for rho in np.array([0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.25,0.5,1.]):
        print
        print
        print 'RHO: ', rho, '  RGC ', i
        reg_L1 = FALMS.L1_L2( rho )
        params = FALMS.falms(params,optimize_L2,reg_L1,ftol=2e-7,callback=falms_callback)
        r['L1'][i][rho] = params
        print [ (rh,par['M'][par['M']<>0].shape[0]) for rh,par in sorted(r['L1'][i].items())]
    save()


plot_vectors([[q['theta'] for q in r['true']      ], [q['theta'] for q in r['ML']           ], \
              [q['theta'] for q in r['analytical']], [q['theta'] for q in r['projected_ML']]])

plot_matrices([r['true'][0]['M'], r['ML'][0]['M'], r['analytical'][0]['M'], r['projected_ML'][0]['M']])

plot_projection_of_U( onto = np.transpose(reinjected_VA[0:Nsub,:]), name='svd(theta,M)', figure=5)
plot_projection_of_U( onto = np.transpose(           VI[0:Nsub,:]), name='svd(STA,STC)', figure=6)