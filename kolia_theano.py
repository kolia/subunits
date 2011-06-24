from inspect import getargspec
from copy    import copy
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


def shapely_tensor( name , x , dtype='float64'):
    if isinstance(x,type(0)):
        return Th.dscalar(name)
    if isinstance(x,type(array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(name),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')


class term:
    '''
    Fields necessary for operating on objectives symbolically,
    and sufficient to build a numerical objective:
    self.theano, self.flatParam, self.Params, self.Args, self.init_params
    '''
    def __init__(self, init_params=None, differentiate=[], **theano_defs):
        names,_,_,defaults = getargspec(theano_defs.itervalues().next())
#        self.defs          = [theano_defs]
        self.Args          = dict(zip(names,defaults))
                
        self.init_params   = self.__intersect_dicts(self.Args.keys(),init_params)
        
        self.Params    = dict((n,shapely_tensor(n,x)) for n,x in self.init_params.items())
        self.flatParam  = Th.concatenate([Th.flatten(x) for _,x in sorted(self.Params.items())])
        for name in sorted(self.init_params.keys()):  del self.Args[name]

        self.__pack   = function([x for _,x in sorted(self.Params.items())],self.flatParam)

        Params_Out = copy(self.Args)
        n = 0
        for name,template in sorted(self.init_params.items()):
            Params_Out[name] = Th.reshape( self.flatParam[n:n+size(template)], template.shape)
            n = n + size(template)

        self.splitParam = \
        function([self.flatParam],[Params_Out[name] for name in sorted(self.Params.keys())])

        def package(some_function):
            def packaged_function(params,arg_dictionary):
                argz = [arg_dictionary[n] for n in sorted(self.Args.keys())]
                return some_function(self.flatten(params),*argz)
            return packaged_function

        arglist = [value for _,value in sorted(self.Args.items())]
        self.theano       = {}
        for name,gen in theano_defs.items(): self.theano[name] = gen(**Params_Out)
        for name in differentiate:
            self.theano['d'+name] = Th.grad( cost              = self.theano[name] , 
                                             wrt               = self.flatParam , 
                                             consider_constant = arglist)

        for name,output in self.theano.items():
            setattr(self,name,package(function([self.flatParam]+arglist,output)))

    def repack(self, dic, ll): return dict((name,ll[i]) for i,name in enumerate(dic.keys()))

    def inflate(self, x):
        if isinstance(x,type(array([]))):
            x = self.splitParam(x)
        if isinstance(x,type([])):
            return self.repack(self.Params,x)
        raise TypeError('expects a numpy ndarray or a list of numpy ndarrays')

    def flatten(self,params):
#        debug_here()
        if isinstance(params,type(dict())):
            params = [x for _,x in sorted(params.items())]
        if isinstance(params,type([])):
            return self.__pack(*params)
        if isinstance(params,type(array([]))):
            return params
        raise TypeError('expects a numpy ndarray or a list of numpy ndarrays')

    def __intersect_dicts(self,names,d):
        out = {}
        for name in names:
            try:              out[name] = d[name]
            except KeyError:  ()
        return out



class sum_objective():
    def __init__(self, l):
        self.terms     = l
        self.Params    = [objective.Params      for objective in l]
        self.Args      = [objective.Args        for objective in l]
        self.initParam = [objective.init_params for objective in l]
        self.flatParam = Th.concatenate([objective.flatParam for objective in l])

        self.joinParam  = function( [ll.flatParam for ll in l] , self.flatParam )

        Params_Out = []
        n = 0
        for init_param in self.initParam:
            dn = sum([size(value) for _,value in sorted(init_param.items())])
            Params_Out = Params_Out + [self.flatParam[n:n+dn]]
            n = n + dn
        self.splitParam = function([self.flatParam], Params_Out )
    
#        names  = set(sum([objective.theano.keys() for objective in l],[]))
#        self.theano = dict((name,0) for name in names)
#        for objective in l:
#            for name in names:
#                try:    self.theano[name] = self.theano[name] + objective.theano[name]
#                except: print 'Error adding ', name, ' to ', objective
#        for name,target in self.theano.items():
#            setattr(self, name, wrap(function([self.flatParam] + self.unpack(self.Args), target)))

        def wrap(name,reduction):
            def wrapped_function(params,arg_dicts):
#                argz = [[d[n] for n in sorted(args.keys())] for args,d in zip(self.Args,arg_dicts)]
                flats = self.splitParam( self.flatten(params) )                
                return reduction([ getattr(ll,name)(x,arg_dict) for ll,x,arg_dict in zip(l,flats,arg_dicts)])
            return wrapped_function

        for name in l[0].theano.keys():
            if (name[0] == 'd') and (name[1:] in l[0].theano):
                setattr(self,name, wrap(name, lambda l : self.joinParam(*l) ))
            else:            
                setattr(self,name, wrap(name, sum))

    def __unpack(self,dict_list):
        return [[value for _,value in sorted(d.items())] for d in dict_list]

    def __repack(self, dict_list, ll):
        result = [{} for d in dict_list]
        n = 0
        for i,d in enumerate(dict_list):
            for name in sorted(d.keys()):
                result[i][name] = ll[n]
                n = n+1
        return result

    def inflate(self,x):
        if isinstance(x,type(array([]))):
            x = self.splitParam(x)
        if isinstance(x,type([])) and isinstance(x[0],type(array([]))):
            x = [term.splitParam(param) for term,param in zip(self.terms,x)]
        if isinstance(x,type([])):
            return self.__repack(self.Params,sum(x,[]))
        raise TypeError('expects a numpy ndarray or a list of numpy ndarrays')

    def flat_list(self,params):
        if isinstance(params,type([])) and isinstance(params[0],type(dict())):
            params = self.__unpack(params)
        if isinstance(params,type([])) and isinstance(params[0],type([])):
            return [term.flatten(pl) for term,pl in zip(self.terms,params)]
        if isinstance(params,type(array([]))):
            return self.splitParam(params)
        raise TypeError('expects a list of dicts, list of lists or numpy ndarray')

    def flatten(self,params):
        if isinstance(params,type(array([]))):
            return params
        return self.joinParam(*self.flat_list(params))
