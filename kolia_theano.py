from inspect import getargspec
from copy    import copy, deepcopy
from numpy   import size, array, asarray
import numpy.linalg
from theano  import function
import theano.tensor  as Th
from theano.gof import Op, Apply
from theano.sandbox.linalg import matrix_inverse

from IPython.Debugger import Tracer; debug_here = Tracer()


class SLogDet(Op):
    """matrix determinant"""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o1 = Th.scalar(dtype=x.dtype)
        o2 = Th.scalar(dtype=x.dtype)
        return Apply(self, [x], [o1,o2])
    def perform(self, node, (x,), (z1,z2, )):
        try:
            s,ldet = numpy.linalg.slogdet(x)
            z1[0] = asarray(s   , dtype=x.dtype)
            z2[0] = asarray(ldet, dtype=x.dtype)
        except:
            print 'Failed to compute determinant', x
            raise
    def grad(self, inputs, g_outputs):
        gz1,gz2 = g_outputs
        x, = inputs
#        debug_here()
        return [gz2 * matrix_inverse(x).T]
    def __str__(self):
        return "SLogDet"
slogdet = SLogDet()


class Eig(Op):
    """matrix eigenvalues and eigenvectors"""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o1 = Th.vector(dtype=x.dtype)
        o2 = Th.matrix(dtype=x.dtype)
        return Apply(self, [x], [o1,o2])
    def perform(self, node, (x,), (z1,z2, )):
        try:
            w,v = numpy.linalg.eig(x)
            z1[0] = asarray(w, dtype=x.dtype)
            z2[0] = asarray(v, dtype=x.dtype)
        except:
            print 'Failed to compute eig', x
            raise
    def __str__(self):
        return "Eig"
eig = Eig()


def shapely_tensor( x , dtype='float64'):
    if isinstance(x,type(0)):
        return Th.dscalar()
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')


class objective:
    '''
    Fields necessary for operating on objectives symbolically,
    and sufficient to build a numerical objective:
    self.theano, self.flatParam, self.Params_In, self.Params_Out, self.Args
    '''
    def __init__(self, theano_objective, init_params=None, **other):
        arg_names,_,_,arg_defaults = getargspec(theano_objective)
        self.theano_objective      = theano_objective
        self.init_params           = init_params
        self.Args                  = dict(zip(arg_names,arg_defaults))
        self.Params_Out            = copy(self.Args)

        self.Params_In = dict((n,shapely_tensor(x)) for n,x in init_params.items())
        self.flatParam  = Th.concatenate([Th.flatten(x) for _,x in sorted(self.Params_In.items())])

        n = 0
        for name,template in sorted(init_params.items()):
            del self.Args[name]
            self.Params_Out[name] = Th.reshape( self.flatParam[n:n+size(template)], 
                                                template.shape)
            n = n + size(template)
        self.theano       = {}
        self.theano['f' ] = theano_objective(**self.Params_Out)
        self.theano['df'] = Th.grad( cost = self.theano['f'] , 
                                     wrt  = self.flatParam , 
                                     consider_constant=[v for _,v in sorted(self.Args.items())])

        for name,gen in other.items(): self.theano[name] = gen(**self.Params_Out)
        self.deploy()

    def deploy(self):
        self.__flat   = function([v for n,v in sorted(self.Params_In.items())] , self.flatParam )
        self.__unflat = function([self.flatParam], [self.Params_Out[n] for n in sorted(self.Params_In.keys())] )

        def package(some_function):
            def packaged_function(params,arg_dictionary):
                argz = [arg_dictionary[n] for n in sorted(self.Args.keys())]
                return some_function(self.flatten(params),*argz)
            return packaged_function

        self.flatAll_In = [self.flatParam] + [v for n,v in sorted(self.Args.items())]
        for name,target in self.theano.items():
            setattr(self, name, package(function( self.flatAll_In,  target )))

    def inflate(self,x):
        if isinstance(x,type(array([]))):
            x = self.__unflat(x)
        if isinstance(x,type([])):
            return dict(zip(sorted(self.Params_In.keys()),x))
             
    def flatten(self,params):
        if isinstance(params,type({})):
#            debug_here()
            params = [params[name] for name in sorted(self.Params_In.keys())]
        if isinstance(params,type([])):
            return self.__flat(*params)
        if isinstance(params,type(array([]))):
            return params
        raise TypeError('flatten expects a dict, list or numpy ndarray')

    def __add__(self,y):
        for name,target in self.theano.items():
            try:    self.theano[name] = self.theano[name] + y.theano[name]
            except: print 'Error adding ', name
        self.Params_In  = dict( self.Params_In.items()  + y.Params_In.items()  )
        self.Params_Out = dict( self.Params_Out.items() + y.Params_Out.items() )
        self.Args       = dict( self.Args.items()       + y.Args.items()       )
        self.flatParam   = Th.concatenate([self.flatParam,y.flatParam])
        self.deploy()
       
    def __rmul__(self,n):
        '''self.theano, self.flatParam, self.Params_In, self.Params_Out, self.Args'''
        replicas = [deepcopy((self.theano,self.Params_In,self.Params_Out,
                              self.Args,self.flatParams)) for _ in range(n)]
        for name in self.theano.keys():
            self.theano[name] = Th.add(*[r[0][name] for r in replicas])        
        def reducer(i):
            new_d = {}
            for j,r in enumerate(replicas):
                new_d.update([((n,j),r[i][n]) for n in replicas[0][i].keys()])
            return new_d
        self.Params_In  = reducer(1)
        self.Params_Out = reducer(2)
        self.Args       = reducer(3)
        self.flatParam = Th.concatenate([r[4] for r in replicas])
        return self.deploy()
        