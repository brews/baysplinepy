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

    n_uk = len(uk)
    n_posterior = len(tau2_draws_final)

    # Nsamps
    n_iter = 500
    burnin = 250

    # Set priors. Use prahl conversion to target mean and std
    prior_mean = np.median((uk - 0.039) / 0.034)

    # Save priors to output
    output['prior_mean'] = prior_mean
    output['prior_std'] = pstd

    # Vectorize priors
    prior_mean = output['prior_mean'] * np.ones(n_uk)
    prior_var = output['prior_std'] ** 2 * np.ones(n_uk)

    # Set an initial SST value
    initial_sst = prior_mean

    mh_samples = np.empty((n_uk, n_posterior, n_iter - burnin))
    mh_samples[:] = np.nan
    accepts_t = np.empty((n_uk, n_posterior, n_iter - burnin))
    accepts_t[:] = np.nan

    # Make a spline with set knots
    degree = 2  # order is 3
    kn = augknt(knots, degree)

    if output['prior_mean'] < 20:
        jump_dist = 3.5
    elif 20 <= output['prior_mean'] <= 23.7:
        jump_dist = 3.7
    else:
        jump_dist = output['prior_mean'] * 0.8092 - 15.1405

    output['jump_dist'] = jump_dist  # Should be 3.5 in test case.

    # MH loop
    for jj in range(n_posterior):

        accepts = np.empty((n_uk, n_iter))
        accepts[:] = np.nan
        samples = np.empty((n_uk, n_iter))
        samples[:] = np.nan

        # Initialize at starting value
        samples[:, 0] = initial_sst
        sample_now = samples[:, 0]

        b_now = b_draws_final[jj, :]
        tau_now = tau2_draws_final[jj]
        # use spmak to put together the bspline
        tck = [kn, b_now, degree]

        # evaluate mean UK value at current SST
        mean_now = interpolate.splev(x=sample_now, tck=tck, ext=0)

        # Evaluate likelihood
        likelihood_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now))

        # Evaluate prior
        prior_now = stats.norm.pdf(sample_now, prior_mean, np.sqrt(prior_var))

        # multiply to get initial proposal S0
        initial_proposal = likelihood_now * prior_now

        for kk in range(1, n_iter):
            # generate proposal using normal jumping distr.
            proposal = np.random.normal(sample_now, jump_dist)
            # evaluate mean value at current sst
            mean_now = interpolate.splev(x=proposal, tck=tck, ext=0)
            # evaluate liklihood
            likelihood_now = stats.norm.pdf(uk, mean_now, np.sqrt(tau_now))
            # evaluate prior
            prior_now = stats.norm.pdf(proposal, prior_mean, np.sqrt(prior_var))
            # multiply to get proposal update_proposal
            update_proposal = likelihood_now * prior_now

            mh_rate = update_proposal / initial_proposal
            success_rate = np.minimum(1, mh_rate)

            # make the draw
            draw = np.random.uniform(size=n_uk)
            b = draw <= success_rate
            sample_now[b] = proposal[b]
            initial_proposal[b] = update_proposal[b]

            accepts[b, kk] = 1
            samples[:, kk] = sample_now
            
        mh_samples[:, jj, :] = samples[:, burnin:]
        accepts_t[:, jj, :] = accepts[:, burnin:]

    # Now let's calculate the rhat statistic to assess convergence
    # TODO(brews): See if we can't clean up the below and just use chaincovergence()
    rhats = np.empty((mh_samples.shape[0], 1))
    for i in range(mh_samples.shape[0]):
        rhats[i], neff = chainconvergence(mh_samples[i, ...].squeeze(), n_posterior)
    output['rhat'] = np.median(rhats, axis=0)

    # reshape
    mh_c = mh_samples.reshape([n_uk, n_posterior * (n_iter-burnin)], order='F')

    # Calculate acceptance
    output['accepts'] = np.nansum(accepts_t) / (n_uk * n_posterior * (n_iter - burnin))

    # Sort and assign to output
    mh_s = mh_c.copy()
    mh_s.sort(axis=1)
    pers5 = np.round(np.array([0.05, 0.16, 0.5, 0.84, 0.95]) * mh_c.shape[1]).astype('int')
    output['sst'] = mh_s[:, pers5]

    # take a subsample of MH to work with for ks.   
    mh_subsample = mh_s[:, 1::50]
    output['ens'] = mh_subsample
    return output
