import numpy

import retina
reload(retina)

# Memoizing results using joblib;  makes life easier
from joblib import Memory
memory = Memory(cachedir='/Users/kolia/Documents/joblibcache', verbose=0)

data = retina.read_data()

cones             = data['cone_locations']
possible_subunits = cones[:100,:]

filters = retina.gaussian2D_weights( cones, possible_subunits, sigma=2. )

#V2 = 0.1
#def NL(x): return x + 0.5 * V2 * ( x ** 2 )


# BREAK THIS INTO GROUPS OF RGCs,  MEMORY BLOWING UP!!!
@memory.cache
def localization( datum ):
    return retina.accumulate_statistics( 
        data_generator = retina.read_stimulus( datum['spikes'],
                                               stimulus_pattern='cone_input_%d.mat' ) )

#data['spikes'] = data['spikes'][:2,:]
stats = localization( data )