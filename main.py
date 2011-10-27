import numpy

import retina
reload(retina)

import IRLS
reload(IRLS)
from   IRLS import IRLS

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

data = retina.read_data()

cones             = data['cone_locations']
possible_subunits = cones

#V2 = 0.1
#def NL(x): return x + 0.5 * V2 * ( x ** 2 )

def rgc_type( i , data=data ):
    eyedee = data['rgc_ids'][i]
    for rgctype, inds in data['rgc_types'].items():
        if eyedee in inds: return rgctype
    return None

filters = retina.gaussian2D_weights( cones, possible_subunits, sigma=3. )

from kolia_base import save
save(data,'data_localization')


@memory.cache
def localization( filters ):
    return retina.accumulate_statistics( 
        data_generator = 
            retina.read_stimulus( data['spikes'],
                                 stimulus_pattern='cone_input_%d.mat' ) ,
        feature= lambda x : numpy.dot(filters,x))

#data['spikes'] = data['spikes'][:2,:]
stats = localization( filters )

import pylab
def hista(i , data=data, stats=stats, N=2):
    pylab.close('all')
    pylab.figure(1, figsize=(10,12))
    for j in range(N): 
        pylab.subplot(N,1,j+1)
        pylab.hist( stats['STA'][i+j] , bins=30)
        pylab.ylabel(rgc_type(i+j,data=data))

from kolia_base import save
save(stats,'Linear_localization')

def n_largest( values , keep=10 ):
    sorted_values = numpy.abs(values)
    sorted_values.sort()
    cutoff = sorted_values[-keep]
    return numpy.nonzero( numpy.abs(values) >= cutoff )[0]
    
@memory.cache
def fit_U_stats( rgc_type='off midget', keep=15, stats=stats ):
    which_rgc = [i for i,ind in enumerate(data['rgc_ids'])
                   if  ind   in data['rgc_types'][rgc_type]]
    spikes = data['spikes'][which_rgc,:]
    sparse_index = [n_largest( sta, keep=keep ) for sta in stats['STA']]
    stats['rgc_type']  = rgc_type
    stats['rgc_index'] = which_rgc
    stats.update( retina.accumulate_statistics( 
        data_generator = retina.read_stimulus( spikes ) ,
        feature        = lambda x : x                   ,
        pipelines      = retina.fit_U                    ,
        sparse_index   = sparse_index                   ))
    return stats

ustats = {}
ustats['off midget'] = fit_U_stats( rgc_type='off midget', keep=20 )

from kolia_base import save
save(ustats,'ustats_0')

import sys

dSTA  = numpy.concatenate(
        [STA[:,numpy.newaxis]-stats['mean'][:,numpy.newaxis] 
         for STA in stats['STA']], axis=1)

print 'calculated dSTA'
sys.stdout.flush()

Cin = stats['cov']/2

print 'calculated Cin'
sys.stdout.flush()

from scipy.linalg   import schur
@memory.cache
def ARD( dSTA , Cin , lam=0.0001 ):
    print 'Starting ARD of size ', Cin.shape,' with lambda=',lam
    sys.stdout.flush()
    D,Z = schur(Cin)
    print 'Schur decomposition completed'
    sys.stdout.flush()
    DD  = numpy.diag(D)
    keep= DD>1e-10
    P   =  (Z[:,keep] * numpy.sqrt(DD[keep])).T
    y   =  numpy.dot ( (Z[:,keep] * 1/numpy.sqrt(DD[keep])).T , dSTA ) / 2    
    iW = 1e-1
    for i in range(100):
        print 'Irlsing'
        sys.stdout.flush()
        V, iW = IRLS( y, P, x=0, disp_every=10, lam=lam, maxiter=200 , 
                      ftol=1e-5, nonzero=1e-1, iw=iW)
        save({'V':V,'iW':iW},'Localizing_lam%.0e'%lam)
    return V, iW
    
V, iW = ARD( dSTA , Cin , lam=0.009 )