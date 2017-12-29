from scipy.io import loadmat
import pystan
finals = loadmat('bayes_posterior.mat')

# need to define `age`, `uk`, `pstd`

b_draws_final = finals['b_draws_final']
tau2_draws_final = finals['tau2_draws_final']
knots = finals['knots']

n_ts = len(uk)
n_p = len(tau2_draws_final)

n=500
burnin = 250

uk_mean = (uk - 0.039)/0.034
uk_sigma = pstd

# bspline function from http://mc-stan.org/users/documentation/case-studies/splines_in_stan.html
stan_bspline = """
functions {
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order);
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order) {
    // INPUTS:
    //    t:          the points at which the b_spline is calculated
    //    ext_knots:  the set of extended knots
    //    ind:        the index of the b_spline
    //    order:      the order of the b-spline
    vector[size(t)] b_spline;
    vector[size(t)] w1 = rep_vector(0, size(t));
    vector[size(t)] w2 = rep_vector(0, size(t));
    if (order==1)
      for (i in 1:size(t)) // B-splines of order 1 are piece-wise constant
        b_spline[i] = (ext_knots[ind] <= t[i]) && (t[i] < ext_knots[ind+1]);
    else {
      if (ext_knots[ind] != ext_knots[ind+order-1])
        w1 = (to_vector(t) - rep_vector(ext_knots[ind], size(t))) /
             (ext_knots[ind+order-1] - ext_knots[ind]);
      if (ext_knots[ind+1] != ext_knots[ind+order])
        w2 = 1 - (to_vector(t) - rep_vector(ext_knots[ind+1], size(t))) /
                 (ext_knots[ind+order] - ext_knots[ind+1]);
      // Calculating the B-spline recursively as linear interpolation of two lower-order splines
      b_spline = w1 .* build_b_spline(t, ext_knots, ind, order-1) +
                 w2 .* build_b_spline(t, ext_knots, ind+1, order-1);
    }
    return b_spline;
  }
}
data {
    int<lower=0> n; // number of measures
    real s_now[n]; // uk observations
    real knots[3];  // bspline knots
}
transformed data {
    real bs[n];
    bs = build_b_spline(s_now, knots, 1, 3);
}
"""

stan_code2 = """
data {
    int<lower=0> n; // number of measures
    real x[n]; // uk observations
    real y[n]; //
}
parameters {
    real mu;
    real<lower=0> sigma;
    real alpha;
}
model {
    y ~ alpha + normal(mu, sigma) * x;
}
"""



# sm = pystan.StanModel(model_code=stan_bspline + '\n' + stan_code2)

sm = pystan.StanModel(model_code=stan_bspline)
fit = sm.sampling(data = {'n': 10, 's_now': list(range(11)), 'knots': [-0.4,  23.6,  29.6]}, iter = 5, algorithm='Fixed_param')
