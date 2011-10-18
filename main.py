import numpy

from scipy.io import loadmat
data = loadmat('data.mat')
data = data['data']

spikes = data['spike_rate'][0][0]

rgc_ids = data['rgc_ids'][0][0][0]

cone_weights = data['cone_weights'][0][0]

cone_types = data['cone_types'][0][0].tolist()

cones = data['cone_locations'][0][0]

rgc_locations = numpy.array([d[0][0] for d in data['rgc_locations'][0][0]])

rgc_types = dict((d[0][0],d[1][0].tolist()) 
                  for d in filter( lambda d : len( d[0] )>0 , [d[0][0] 
                  for d in data['cell_types'][0][0][0]] ))


import simulate_retina
reload(simulate_retina)
from simulate_retina import *
V2 = 0.1
def NL(x): return x + 0.5 * V2 * ( x ** 2 )

S = 10

possible_subunits = hexagonal_2Dgrid( spacing=1. , field_size_x=S , field_size_y=S ) 

filters = gaussian2D_weights( cones , possible_subunits , 
                             sigma=2. )
