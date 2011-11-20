import kolia_base as kb
import copy
import numpy
import numpy.linalg
from theano  import function
import theano.tensor  as Th
from theano.gof import Op, Apply
from theano.sandbox.linalg import matrix_inverse

#from IPython.Debugger import Tracer; debug_here = Tracer()

class LogDet(Op):
    """matrix determinant
    TODO: move this op to another file that request scipy.
    """
    def make_node(self, x):
        x = Th.as_tensor_variable(x)
        o = Th.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            s,ldet = numpy.linalg.slogdet(x)
            z[0] = numpy.asarray(ldet, dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise
    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * matrix_inverse(x).T]
    def __str__(self):
        return "Det"
logdet = LogDet()

class Eig(Op):
    """SYMBOLIC matrix eigenvalues and eigenvectors, without gradient."""
    def make_node(self, x):
        x  = Th.as_tensor_variable(x)
        o1 = Th.vector(dtype=x.dtype)
        o2 = Th.matrix(dtype=x.dtype)
        return Apply(self, [x], [o1,o2])
    def perform(self, node, (x,), (z1,z2, )):
        try:
            w,v = numpy.linalg.eig(x)
            z1[0] = numpy.asarray(w, dtype=x.dtype)
            z2[0] = numpy.asarray(v, dtype=x.dtype)
        except:
            print 'Failed to compute eig', x
            raise
    def __str__(self):
        return "Eig"
eig = Eig()

def reown( x , y ):
    x.owner = y.owner
    if x.owner is not None:
        for i,output in enumerate( x.owner.outputs ):
            if output is y: x.owner.outputs[i] = x

def shapely_tensor( name , x , dtype='float64'):
    '''Return SYMBOLIC tensor with the same dimensions and size as input.'''
    if isinstance(x,type(0)):
        return Th.dscalar(name)
    if isinstance(x,type(numpy.array([]))):
        dtensor_x = Th.TensorType(dtype, (False,)*x.ndim)
        return Th.specify_shape(dtensor_x(name),x.shape)
    raise TypeError('shapely_tensor expects a scalar or numpy ndarray')

def _dict2list( X ):
    if type(X) is type({}):
        return [x for _,x in sorted( X.items())]
    return X

def list2dict( _list, template ):
    if type(template) is type({}):
        return dict(zip(sorted( template.keys()), _list))
    return _list

def simplify( inputs , outputs ):
    input_list, output_list = _simplify( _dict2list(inputs), _dict2list(outputs) )
    return list2dict( input_list, inputs), list2dict( output_list, outputs)

def _simplify( inputs , outputs ):
    env = simplified_env( inputs, outputs )
    new_outputs = env.outputs
    env.disown()
    return inputs , new_outputs

def simplified_env( inputs, outputs ):
    return reconnect_env( inputs,  
                function( _dict2list(inputs), _dict2list(outputs) ).maker.env)

from theano.gof.env import Env
def make_env( inputs, outputs ):
    return Env( _dict2list(inputs), _dict2list(outputs) )

def reconnect_env( inputs, env ):
    old_inputs  = env.inputs
    inputs = _dict2list(inputs)
    env.inputs = inputs
    for ni,i in zip(old_inputs, inputs ):
        if ni is not i:
            env.replace(ni,i)
    return env

class Params( object ):
    def __init__(self, example=None):
        self.Example_Params = example
        params = dict((n,shapely_tensor(n,x)) for n,x in example.items())
        self.flatParam  = Th.concatenate([Th.flatten(x) for _,x in sorted(params.items())])

        n = 0
        self.Params = {}
        for name,template in sorted(example.items()):
            self.Params[name] = Th.reshape( self.flatParam[n:n+numpy.size(template)], 
                                            template.shape)
            n = n + numpy.size(template)

    def flat(self,X): return kb.flat(X)
    def unflat(self,X): return kb.unflat(self.Example_Params,X)


class Objective( Params ):
    '''Compile a list of theano expression generators given as named 
    arguments into an object with corresponding numerical functions, as a 
    function of the parameters in dictionary init_params.  All of these 
    functions can be transparently called either with parameters structured 
    as nested dictionaries and lists of numpy arrays, of by a flat vector of
    appropriate length concatenating all the parameters.
    
    Init_params only serves as a template for determining which variables
    are required, and their types and sizes are.
    All other dependencies of the theano targets should be provided as 
    arguments using objective.where(), which spawns off an objective with 
    the values of the arguments fixed to the specified values.

    For all parameter names appearing in init_params that are in the list 
    differentiate, a numerical function returning the gradient with respect 
    to parameters of that quantity will be provided, under the name 
    'd' + name; for example, the gradient of f will become objective.df

    Optionally, a callback function of the parameters'''
    def __init__(self, params, init_params , args, outputs, 
                 differentiate=[], mode='FAST_RUN'):
        # Set up parameters
        super(Objective,self).__init__( example=init_params )
        for name,p in params.items():
            reown(p,self.Params[name])

        # Set up list of constant arguments
        self.Args    = dict([(n,Th.as_tensor_variable(d,name=n)) for n,d in args.items()])        
        self.ArgList = [arg for _,arg in sorted(self.Args.items())]

        # Differentiate objective function
        self.outputs       = outputs
        self.differentiate = differentiate
        for name in self.differentiate:
            if ('d'+name) not in self.outputs:
                self.outputs['d'+name] = Th.grad( cost              = outputs[name] ,
                                                  wrt               = self.flatParam ,
                                                  consider_constant = self.ArgList)

        # Have theano compile actual objective functions
        self.functions = {}
        for name, output in self.outputs.items():
            self.functions[name] = function([self.flatParam]+self.ArgList,
                                             output,mode=mode,accept_inplace=True)

    def where(self,other,**args):
        t = copy.copy(self) #object()   # replace with deepcopy of self?
        for name,value in other.items():
            setattr(t,name,value)
        t.ArgValues = [args[n] for n in sorted(self.Args.keys())]
#        print 'Objective.where: the following arguments have been fixed:'
#        print [(n,args[n].shape) for n in sorted(self.Args.keys())]
        def package(some_function):
            def packaged_function(params): return some_function(kb.flat(params),*t.ArgValues)
            return packaged_function
        for name in self.outputs.keys():
            setattr(t,name,package(self.functions[name]))
        def with_callback(callbk):
            def callback(params,**kwargs): return callbk(t,params,**kwargs)
            t.callback = callback
            return t
        t.with_callback = with_callback
        return t
        


#        self.splitParam = \
#        function([self.flatParam],[self.Params_Out[name] for name in sorted(self.Params.keys())])

#        for name in self.hessian:
#            if ('H'+name) not in self.theano_functions and name in self.theano:
#                gen = self.theano[name]
#                direction = 
#                self.theano_functions['H'+name] = function( [self.flatParam]+self.arglist) ,
#                    Hessian_along( gen(self.theano[name]) ,  )

#    def __get_state__(self):
#        return (Args)
##        return (Args, theano_functions)
#
#    def __set_state__(self,state):
#        (Args) = state
##        (Args, theano_functions) = state
#        self.Args             = Args
##        self.theano_functions = theano_functions
##        self.theano           = theano

#def compose( g , f , connector=None):
#    if connector is None:
#        connection    = []
#        g_inputs      = dict((inpt,None) for inpt in g.inputs)
#        g_input_names = {}
#        g = g.maker.env
#        f = f.maker.env
#        for variable in g.input_names:
#            if g_input_names.has_key(variable.name):
#                print 'g has more than one variable with the same name in compose(g,f)'
#            g_input_names[variable.name] = variable
#        for out in f.outputs:
#            if out.name is None:
#                raise ValueError('outputs must be named in compose(g,f,connector=None)')
#            try:
#                connection += [(out,g_inputs[out.name])]
#            except KeyError:
#                raise ValueError('compose(g,f): name of output of f not found in input of g')
#    for new_r, r in connection:
#        g.replace( r, new_r )
#        del g_inputs[r]
#    f.disown()
#    g.disown()
