import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate

from bayspline.posterior import draws
from bayspline.utils import chainconvergence, augknt


def predict_uk(age, sst):
    """Predict a UK'37 value given SST

    Parameters
    ----------
    age : 1d array_like
        Indicates the age of each element in `uk`. Array length is N.
    sst : 1d array_like
        SST values. Array length is N.

    Returns
    -------
    output : dict

        uk : ndarry
            1500 x N array of inferred ensemble UK'37 values
    """
    output = dict()

    b_draws_final = draws['b_draws_final']
    tau2_draws_final = draws['tau2_draws_final']
    knots = draws['knots'].ravel()

    output['age'] = np.array(age)
    xnew = np.array(sst)

    degree = 2

    tck = [augknt(knots, degree), b_draws_final, degree]
    mean_now = interpolate.splev(x=xnew, tck=tck, ext=0)
    ynew = np.random.normal(mean_now, np.sqrt(tau2_draws_final))
    ynew = ynew.T

    output['uk'] = ynew
    return output


def predict_sst(age, uk, pstd):
    """Predict SST value given UK'37

    Parameters
    ----------
    age : 1d array_like
        Indicates the age of each element in `uk`. Array length is N.
    uk : 1d array_like
        UK'37 values. Array length is N.
    pstd : float
        Prior standard deviation. Recommended values are 7.5 - 10 for most
        UK'37 data. Lower values are usually fine for UK'37 data with a
        smaller range.

    Returns
    -------
    output : dict
        prior_mean : float
            Prior mean value, taken from the mean of the UK'37 data converted
            to SST with the Prahl equation.
        prior_std : float
            Prior standard deviation (set by user).
        jump_dist : float
            Standard deviation of the jump distribution. Values are chosen
            to achieve an acceptance rate of ca. 0.44 [1]_.
        sst : ndarry
            5 x N array of inferred SSTs, includes 5% level (lower 2sigma),
            16% level (lower 1sigma), 50% level (median values),
            84% level (upper 1sigma), and 95% level (upper 2 sigma).

    References
    ----------
    .. [1] Gelman, Andrew, ed. Bayesian Data Analysis. 2nd ed. Texts in
        Statistical Science. Boca Raton, Fla: Chapman & Hall/CRC, 2004.
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

    # Nsamps
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
    degree = 2  # order is 3
    kn = augknt(knots, degree)

    if pm < 20:
        jw = 3.5
    elif 20 <= pm <= 23.7:
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
            # evaluate mean value at current sst
            mean_now = interpolate.splev(x=s_prop, tck=tck, ext=0)
            # evaluate liklihood
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
