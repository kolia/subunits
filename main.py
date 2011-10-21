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

filters = retina.gaussian2D_weights( cones, possible_subunits, sigma=3. )

from kolia_base import save
save(data,'data_localization')


# BREAK THIS INTO GROUPS OF RGCs,  MEMORY BLOWING UP!!!
@memory.cache
def localization( datum, filters ):
    return retina.accumulate_statistics( 
        data_generator = 
            retina.read_stimulus( datum['spikes'],
                                 stimulus_pattern='cone_input_%d.mat' ) ,
        feature= lambda x : numpy.dot(filters,x))

#data['spikes'] = data['spikes'][:2,:]
stats = localization( data, filters )

#from kolia_base import save
#save(stats,'Linear_localization')

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
def ARD( dSTA , Cin , lam=0. ):
    print 'Starting ARD of size', Cin.shape
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
        save({'V':V,'iW':iW},'Localizing')
    return V, iW
    
V, iW = ARD( dSTA , Cin , lam=0.0 )