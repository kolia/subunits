from inspect import getargspec
from copy    import copy
from numpy   import size, array, asarray
from numpy.linalg import slogdet
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
            s,ldet = slogdet(x)
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
logdet = SLogDet()


def shapely_tensor( x ):
    if isinstance(x,type(0)):
        return Th.dscalar()
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType('float64', (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')

class objective:
    def __init__(self, theano_objective, init_params=None, **other):
        arg_names,_,_,arg_defaults = getargspec(theano_objective)
#        self.args        = [v for n,v in sorted(args.items(), key=lambda (k,_): k)]
#        self.arg_names   = [n for n,v in sorted(args.items(), key=lambda (k,_): k)]
        self.theano_objective = theano_objective
        self.param_templates = init_params

        argdict = {}
        for arg_name,arg_value in zip(arg_names,arg_defaults):
            argdict[arg_name] = arg_value
        self.args = copy(argdict)
        
        # self.params unpacks a flat vector into a dictionary of parameters
        n = 0
        params  = Th.dvector()
        paramlist = []
        for name,template in sorted(init_params.items()):
            shapely_template = shapely_tensor(template)
            paramlist = paramlist + [shapely_template]
            self.args[name] = Th.reshape( params[n:n+size(template)], template.shape)
            del argdict[name]
            n = n + size(template)

        argnames  = [name for name,value in sorted(argdict.items())]
        argvalues = [value for name,value in sorted(argdict.items())]
        flat_args  = [params] + argvalues
        f  = theano_objective(**self.args)
        df = Th.grad( cost = f , wrt = params , consider_constant=argvalues)

        flatter       = Th.concatenate([Th.flatten(x) for x in paramlist])
        self.__flat   = function(paramlist ,flatter )
        paramlist2   = [self.args[n] for n in sorted(init_params.keys())]
#        unflat        = Th.concatenate([Th.flatten(x) for x in paramlist2])
        self.__unflat = function([params],paramlist2)

        def package(some_function):
            def packaged_function(params,arg_dictionary):
                argz = [arg_dictionary[n] for n in argnames]
                return some_function(self.flatten(params),*argz)
            return packaged_function

        self.init_params = flatter
        self.f      = package(function( flat_args,  f ))
        self.df     = package(function( flat_args, df ))
        for name,gen in other.items():
            setattr(self, name, package(function( flat_args,  gen(**self.args) )))
            
    def inflate(self,x):
        if isinstance(x,type(array([]))):
            x = self.__unflat(x)
        if isinstance(x,type([])):
            return dict(zip(sorted(self.param_templates.keys()),x))
            
    def flatten(self,params):
        if isinstance(params,type({})):
#            debug_here()
            params = [params[name] for name in sorted(self.param_templates.keys())]
        if isinstance(params,type([])):
            return self.__flat(*params)
        if isinstance(params,type(array([]))):
            return params
        raise TypeError('flatten expects a dict, list or numpy ndarray')

