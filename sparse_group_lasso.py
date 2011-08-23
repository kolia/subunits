from numpy import dot, minimum, maximum

def sparsify_coeff(predictor,group_weight,weight,r,coeff):
    return coeff, r-prediction

def sparsify_group(predictor, group_weight, weight, r, coeff):
    if coeff[0] is None:
        residual = r[0]                
    else:
        residual = r[0] + dot(predictor,coeff[0])
    a = dot(predictor,residual)
    t = maximum( minimum( a / weight , 1 ) , -1 )
    J = sum( (a - weight * t)**2 ) / group_weight**2
    if J<=1:
        r[0]     = residual
        coeff[0] = None

def initialize_group_lasso(predictors, group_weights, weights, y, coeffs = None):
    if coeffs is None: coeffs = [[None] for _ in weights]
    r = [y - [dot(predictor,c[0]) for c,predictor in 
              filter(lambda c,_ : c[0] is not None, zip(coeffs,predictors))]]
    return r, coeffs

def group_lasso(predictors, group_weights, weights, r, coeffs):
    for predictor , group_weight , weight , coeff  in \
    zip(predictors, group_weights, weights, coeffs ):
        sparsify_group(predictor,group_weight,weight,r,coeff)