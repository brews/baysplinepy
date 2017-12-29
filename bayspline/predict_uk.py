#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.interpolate as interpolate
from bayspline.posterior import draws


# Variables for test case
uk = np.array([0.4, 0.5, 0.6])
age = np.array([1, 2, 3])
pstd = 7.5


def chainconvergence(chains, m):
    """
    From Jess Tierney's "ChainConvergence.m"
    %
    % function [Rhat, Neff]=ChainConvergence(chains, M)
    % 
    % calculate the R-hat stat form "Bayesian Data Analysis" (page 297) for
    % monitoring  convergence of multiple parallel MCMC runs. Also outputs
    % n_eff from page 298. 
    % chains: a matrix of MCMC chains, each of the same length. 
    % M: the number of different chains - must be one of the dimensions of
    % chains. 
    %
    %test
    %chains=randn(100,50);
    %chains(:,1)=chains(:,1)+10;
    %M=5;
    """
    chains = np.array(chains)
    m = int(m)
    assert m in chains.shape
    if m == chains.shape[0]:
        chains = chains.T
    n = chains.shape[0]

    psi_bar_dot_j = np.mean(chains, axis=0)
    psi_bar_dot_dot = np.mean(psi_bar_dot_j)

    b = (n / (m - 1)) * np.sum((psi_bar_dot_j - psi_bar_dot_dot) ** 2)
    s2_j = (1 / (n - 1)) * np.sum((chains - np.kron(psi_bar_dot_j, np.ones((n, 1)))) ** 2, axis=0)
    w = (1 / m) * np.sum(s2_j)

    var_hat_pos = ((n - 1) / n) * w + (1 / n) * b

    rhat = np.sqrt(var_hat_pos/w)
    neff = m * n * np.min([var_hat_pos / b, 1])
    return rhat, neff


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
    %
    %INPUTS:
    %uk = uk37' values of length N
    %pstd = prior standard deviation. Recommended values are 7.5-10 for most uk
    %timeseries. Lower values OK for timeseries with smaller range.

    %OUTPUTS:
    %output.prior_mean = Prior mean value, taken from the mean of the UK timeseries
    %converted to SST with the Prahl equation.
    %
    %output.prior_std = Prior standard deviation (user set).
    %
    %output.jump_dist = standard deviation of the jumping distribution.
    %Values are chosen to achieve a acceptance rate of ca. 0.44
    %(Gelman, 2003).
    %
    %output.SST = 5 x N vector of inferred SSTs, includes 5% level (lower 2sigma), 16% level
    %(lower 1sigma), 50% level (median values), 84% level (upper 1sigma), and
    %95% level (upper 2 sigma).
    """
    # TODO: Add limit to uk range -- I can make strange numbers with large uk vals (e.g. 28)
    output = dict()
    # draws = loadmat('bayes_posterior.mat')
    b_draws_final = draws['b_draws_final'][::3, :]
    tau2_draws_final = draws['tau2_draws_final'][::3, :]
    knots = draws['knots'].ravel()

    output['age'] = np.array(age)
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
    prior_mean = output['prior_mean'] * np.ones((n_ts))
    prior_var = output['prior_std'] ** 2 * np.ones((n_ts))

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

    output['jump_dist'] = jw  # Should be 3.5 in test case.

    # MH loop
    for jj in range(n_p):

        accepts = np.empty((n_ts, n))
        accepts[:] = np.nan
        samps = np.empty((n_ts, n))
        samps[:] = np.nan

        # Initialize at starting value
        samps[:, 0] = init
        s_now = samps[:, 0]

        b_now = b_draws_final[jj, :]
        tau_now = tau2_draws_final[jj]
        # use spmak to put together the bspline
        tck = [kn, b_now, degree]

        # evaluate mean UK value at current SST
        mean_now = interpolate.splev(x=s_now, tck=tck, ext=0)
        # mean_now should be [0.5125, 0.5125, 0.5125] in test case
        # Evaluate liklihood
        ll_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now))
        # ll_now should be [0.5877, 7.8648, 1.6623]
        # Evaluate prior
        pr_now = stats.norm.pdf(s_now, prior_mean, np.sqrt(prior_var))

        # multiply to get initial proposal S0
        s0_now = ll_now * pr_now  # should be [1.1723; 15.6880; 3.3158]

        for kk in range(1, n):
            # generate proposal using normal jumping distr.
            s_prop = np.random.normal(s_now, jw)
            #evaluate mean value at current sst
            mean_now = interpolate.splev(x=s_prop, tck=tck, ext=0)
            #evaluate liklihood
            ll_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now))
            # evaluate prior
            pr_now = stats.norm.pdf(s_prop, prior_mean, np.sqrt(prior_var))
            # multiply to get proposal s0_p
            s0_p = ll_now * pr_now

            mh_rat = s0_p / s0_now
            success_rate = np.min([1, mh_rat.min()])

            # make the draw
            draw = np.random.uniform(size=n_ts)
            b = draw <= success_rate
            s_now[b] = s_prop[b]
            s0_now[b] = s0_p[b]

            accepts[b, kk] = 1
            samps[:, kk] = s_now
            
        mh_samps_t[:, jj, :] = samps[:, burnin:]
        accepts_t[:, jj, :] = accepts[:, burnin:]

    # Now let's calculate the rhat statistic to assess convergence
    rhats = np.empty((mh_samps_t.shape[0], 1))
    for i in range(mh_samps_t.shape[0]):
        rhats[i], neff = chainconvergence(mh_samps_t[i, ...].squeeze(), n_p)

    output['rhat'] = np.median(rhats, axis=0)
    # reshape
    mh_c = mh_samps_t.reshape([n_ts, n_p * (n-burnin)], order='F')
    # Calculate acceptance
    output['accepts'] = np.nansum(accepts_t) / (n_ts * n_p * (n - burnin))  # TODO: needs testing.

    # Sort and assign to output
    mh_s = mh_c.copy()
    mh_s.sort(axis=1)
    pers5 = np.round(np.array([0.05, 0.16, 0.5, 0.84, 0.95]) * mh_c.shape[1]).astype('int')
    output['sst'] = mh_s[:, pers5]

    # take a subsample of MH to work with for ks.   
    mh_sub = mh_s[:, 1::50]
    output['ens'] = mh_sub
    return output


def plot_outputs_priorpost(d, ax=None):
    if ax is None:
        ax = plt.gca()

    xt = np.arange(0, 40.1, 0.1)
    prior = stats.norm.pdf(xt, d['prior_mean'], d['prior_std'])
    kde = stats.gaussian_kde(d['ens'].flat)
    post = kde(xt)

    ax.plot(xt, prior, 'k--', label='Prior')
    ax.plot(xt, post, 'b-', label='Posterior')
    ax.legend()

    return ax


def plot_outputs_timeseries(d, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(d['age'], d['sst'][:, 2], color='black', linewidth=2)
    ax.plot(d['age'], d['sst'][:, 1], color=(0.4, 0.4, 0.4), linestyle='--', linewidth=1)
    ax.plot(d['age'], d['sst'][:, 3], color=(0.4, 0.4, 0.4), linestyle='--', linewidth=1)

    return ax
