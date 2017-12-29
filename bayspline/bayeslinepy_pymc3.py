import theano
import theano.tensor as TT
import pymc3 as pm
import scipy.optimize
import matplotlib.pyplot as plt



def build_B_spline_deg_zero_degree_basis_fns(breaks, x):
    """Build B spline 0 order basis coefficients with knots at 'breaks'. 
    N_{i,0}(x) = { 1 if u_i <= x < u_{i+1}, 0 otherwise }
    """
    expr = []
    expr.append(TT.switch(x<breaks[1], 1, 0))
    for i in range(1, len(breaks)-2):
        l_break = breaks[i]
        u_break = breaks[i+1]
        expr.append(
            TT.switch((x>=l_break)&(x<u_break), 1, 0) )
    expr.append( TT.switch(x>=breaks[-2], 1, 0) )
    return expr



def build_B_spline_higher_degree_basis_fns(
        breaks, prev_degree_coefs, degree, x):
    """Build the higer order B spline basis coefficients
    N_{i,p}(x) = ((x-u_i)/(u_{i+p}-u_i))N_{i,p-1}(x) \
               + ((u_{i+p+1}-x)/(u_{i+p+1}-u_{i+1}))N_{i+1,p-1}(x)
    """
    assert degree > 0
    coefs = []
    for i in range(len(prev_degree_coefs)-1):
        alpha1 = (x-breaks[i])/(breaks[i+degree]-breaks[i]+1e-12)
        alpha2 = (breaks[i+degree+1]-x)/(breaks[i+degree+1]-breaks[i+1]+1e-12)
        coef = alpha1*prev_degree_coefs[i] + alpha2*prev_degree_coefs[i+1]
        coefs.append(coef)
    return coefs

def build_B_spline_basis_fns(breaks, max_degree, x):
    curr_basis_coefs = build_B_spline_deg_zero_degree_basis_fns(breaks, x)
    for degree in range(1, max_degree+1):
        curr_basis_coefs = build_B_spline_higher_degree_basis_fns(
            breaks, curr_basis_coefs, degree, x)
    return curr_basis_coefs

def step_fn(breaks, intercepts, x):
    basis_fns = build_B_spline_deg_zero_degree_basis_fns(breaks, x)
    f = 0
    for i, basis in enumerate(basis_fns):
        f += intercepts[i]*basis
    return f

def spline_fn_expr(breaks, intercepts, degree, x):
    basis_fns = build_B_spline_basis_fns(breaks, degree, x)
    f = 0
    for i, basis in enumerate(basis_fns):
        f += intercepts[i]*basis
    return f

def logistic(x):
    return 1/(1+np.exp(-x))

def build_loss_fn(data, n_bins, degree):
    breaks = np.histogram(data, n_bins)[1][1:-1]
    for i in range(degree+1):
        breaks = np.insert(breaks, 0, data.min()-1e-6)
        breaks = np.append(breaks, data.max()+1e-6)
    
    xs = TT.vector(dtype=theano.config.floatX)
    ys = TT.vector(dtype=theano.config.floatX)
    intercepts = TT.vector(dtype=theano.config.floatX)

    f = theano.function(
        [intercepts, xs],
        spline_fn_expr(breaks, intercepts, degree, xs)
    )
    
    loss_fn = theano.function(
        [intercepts, xs, ys],
        ((ys - spline_fn_expr(breaks, intercepts, degree, xs))**2).sum() )

    return f, loss_fn

def main():
    """Create and visualize a b-spline curve. 
    This is all example code for how to implemnt spline regression using theano.
    I hope to expand this code to use directly as a node in a neural network.  
    """
    xs = np.arange(-6,6,0.01)
    #ys = np.sin(xs/2)
    ys = logistic(xs)
    n_bins = 3
    degree = 1
    f, theano_loss = build_loss_fn(xs, n_bins=n_bins, degree=degree)
    def loss(theta):
        rv = theano_loss(theta, xs, ys)
        return rv
    
    from scipy.optimize import minimize
    x0 = np.ones(n_bins+degree+1)
    rv = minimize(loss, x0)
    plt.plot(xs, f(rv.x, xs))
    plt.plot(xs, ys)
    plt.show()


# For uk37 psm, degree=2