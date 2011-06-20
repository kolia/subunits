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



def shapely_tensor( x ):
    if isinstance(x,type(0)):
        return Th.dscalar()
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType('float64', (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')

def replica(objective,n):
    objectives = [deepcopy(objective) for i in range(n)]
    

class objective:
    def __init__(self, theano_objective, init_params=None, **other):
        arg_names,_,_,arg_defaults = getargspec(theano_objective)
#        self.dict_ParamsArgs        = [v for n,v in sorted(args.items(), key=lambda (k,_): k)]
#        self.arg_names   = [n for n,v in sorted(args.items(), key=lambda (k,_): k)]
        self.theano_objective = theano_objective
        self.param_templates = init_params

        argdict = {}
        for arg_name,arg_value in zip(arg_names,arg_defaults):
            argdict[arg_name] = arg_value
        self.dict_ParamsArgs = copy(argdict)
        
        # self.params unpacks a flat vector into a dictionary of parameters
        n = 0
        self.flatParam_In   = Th.dvector()
        self.listParams_In = []
        for name,template in sorted(init_params.items()):
            shapely_template = shapely_tensor(template)
            self.listParams_In = self.listParams_In + [shapely_template]
            self.dict_ParamsArgs[name] = Th.reshape( self.flatParam_In[n:n+size(template)], template.shape)
            del argdict[name]
            n = n + size(template)

        self.nameArgs  = [name  for name,value in sorted(argdict.items())]
        self.valueArgs = [value for name,value in sorted(argdict.items())]
        self.flatParam_Args_In = [self.flatParam_In] + self.valueArgs
        self.theano_f  = theano_objective(**self.dict_ParamsArgs)
        self.theano_df = Th.grad( cost = self.theano_f , wrt = self.flatParam_In , consider_constant=self.valueArgs)

        self.flatParam_Out   = Th.concatenate([Th.flatten(x) for x in self.listParams_In])
        self.__flat          = function(self.listParams_In , self.flatParam_Out )
        self.listParams_Out = [self.dict_ParamsArgs[n] for n in sorted(init_params.keys())]
        self.__unflat        = function([self.flatParam_In],self.listParams_Out)

        def package(some_function):
            def packaged_function(params,arg_dictionary):
                argz = [arg_dictionary[n] for n in self.nameArgs]
                return some_function(self.flatten(params),*argz)
            return packaged_function

        self.init_params = self.flatParam_Out
        self.f      = package(function( self.flatParam_Args_In, self.theano_f  ))
        self.df     = package(function( self.flatParam_Args_In, self.theano_df ))
        for name,gen in other.items():
            setattr(self, name, package(function( self.flatParam_Args_In,  gen(**self.dict_ParamsArgs) )))
            
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

