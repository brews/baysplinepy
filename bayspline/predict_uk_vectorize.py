#! /usr/bin/env python

import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
from scipy.io import loadmat


def augknt(knots, degree):
    """Augment knots to meet boundary conditiions

    Python version of MATLAB's augknt().
    """
    # a = []
    # Below is poorly done.
    heads = [knots[0]] * degree
    tails = [knots[-1]] * degree
    return np.concatenate([heads, knots, tails])


def predict_uk(age, uk, pstd):
    """Blah blah blah
    """
    # TODO: Add limit to uk range -- I can make strange numbers with large uk vals (e.g. 28)
    output = dict()
    draws = loadmat('bayes_posterior.mat')
    b_draws_final = draws['b_draws_final']  # TODO: needs to be '(1:3:end, :)'
    tau2_draws_final = draws['tau2_draws_final']  # TODO: needs to be '(1:3:end, :)'
    knots = draws['knots'].ravel()

    uk = np.array(uk)

    n_ts = len(uk)
    n_p = len(tau2_draws_final)

    #Nsamps
    n = 500
    burnin = 250

    # Set priors. Use prahl conversion to target mean and std
    pm = np.median((uk - 0.039) / 0.034)

    # Save priors to output
    output['prior_mean'] = pm
    output['prior_std'] = pstd

    # Vectorize priors
    prior_mean = output['prior_mean'] * np.ones(n_ts)
    prior_var = output['prior_std'] ** 2 * np.ones(n_ts)

    # Set an initial SST value
    init = pm

    mh_samps_t = np.empty((n_ts, n_p, n - burnin))
    mh_samps_t[:] = np.nan
    accepts_t = np.empty((n_ts, n_p, n - burnin))
    accepts_t[:] = np.nan

    # Make a spline with set knots
    degree = 2 # order is 3
    kn = augknt(knots, degree)

    if pm < 20:
        jw = 3.5
    elif pm >= 20 and pm <=23.7:
        jw = 3.7
    else:
        jw = pm * 0.8092 - 15.1405

    output['jump_dist'] = jw

    # MH loop

    accepts = np.empty((n_ts, n))
    accepts[:] = np.nan
    samps = np.empty((n_ts, n))
    samps[:] = np.nan

    # Initialize at starting value
    samps[:, 0] = init
    s_now = samps[:, 0]

    b_now = b_draws_final
    tau_now = tau2_draws_final
    # use spmak to put together the bspline
    tck = [knots, b_now, degree]

    # evaluate mean UK value at current SST
    mean_now = np.array(interpolate.splev(x=s_now, tck=tck, ext=0))
    # Evaluate liklihood
    ll_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now))
    # Evaluate prior
    pr_now = stats.norm.pdf(s_now, prior_mean, np.sqrt(prior_var))

    # multiply to get initial proposal S0
    s0_now = ll_now * pr_now

    for kk in range(1, n):
        # generate proposal using normal jumping distr.
        s_prop = np.random.normal(s_now, jw, (n, s_now.shape[0]))
        #evaluate mean value at current sst
        mean_now = np.array(interpolate.splev(x=s_prop, tck=tck, ext=0))
        #evaluate liklihood
        ll_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now)[:, np.newaxis])
        # evaluate prior
        pr_now = stats.norm.pdf(s_prop, prior_mean, np.sqrt(prior_var))
        # multiply to get proposal s0_p
        s0_p = ll_now * pr_now

        mh_rat = s0_p / s0_now[:, np.newaxis]
        success_rate = np.min([1, mh_rat.min(axis = (0, 2))])

        # make the draw
        draw = np.random.uniform(n_ts, 1)
        b = draw <= success_rate
        # Hack around weird matlab boolen indexing
        # TODO: This needs work
        if b:
            b = 1
        else:
            b = 0

        ######################################## Stopped here. # Have not tested numbers for any of this code.
        s_now = s_prop[:, b]
        s0_now = s0_p[..., b]

        accepts[b, kk] = 1
        samps[:, kk] = s_now
        
    mh_samps_t = samps[:, burnin:]
    accepts_t = accepts[:, burnin:]

    # Now let's calculate the rhat statistic to assess convergence
    # rhats = np.empty((mh_samps_t.shape[0], 1))
    # for i in range(mh_samps_t.shape[0]):
        # TODO: See Jess' rhat code sent to me via email

    # output['rhat'] = np.median(rhats)
    output['rhat'] = None
    ####################################################STOPPED HERE
    # reshape
    # mh_c = np.arange(mh_samps_t).reshape(-1, n_ts, n_p * (n-burnin))  # TODO: needs testing.
    # Calculate acceptance
    # output['accepts'] = np.nansum(accepts_t) / (n_ts * n_p * (n - burnin))  # TODO: needs testing.

    # Sort and assign to output
    # mh_s = mh_c.sort(axis=1)
    # pers5 = np.round([0.05, 0.16, 0.5, 0.84, 0.95] * mh_c.shape(axis=1))
    # output['sst'] = mh_s[:, pers5]

    # Continue on line 132    


