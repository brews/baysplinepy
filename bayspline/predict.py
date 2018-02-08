import attr
import attr.validators as av
import numpy as np
from tqdm import tqdm

from bayspline.modelparams import get_draws
from bayspline.utils import chainconvergence, augknt, extrapolate_spline


@attr.s()
class Prediction:
    """MCMC prediction

    Parameters
    ----------
    ensemble : ndarray
        Ensemble of predictions. A 2d array (nxm) for n predictands and m
        ensemble members.
    prior_mean : float or None, optional
        Prior mean used for the prediction.
    prior_std : float or None, optional
        Prior sample standard deviation used for the prediction.
    """
    ensemble = attr.ib(validator=av.optional(av.instance_of(np.ndarray)))
    prior_mean = attr.ib(default=None)
    prior_std = attr.ib(default=None)

    def percentile(self, q=None, interpolation='nearest'):
        """Compute the qth ranked percentile from ensemble members

        Parameters
        ----------
        q : float ,sequence of floats, or None, optional
            Percentiles (i.e. [0, 100]) to compute. Default is 5%, 50%, 95%.
        interpolation : str, optional
            Passed to numpy.percentile. Default is 'nearest'.
        Returns
        -------
        perc : ndarray
            A 2d (nxm) array of floats where n is the number of predictands in
            the ensemble and m is the number of percentiles ('len(q)').
        """
        if q is None:
            q = [5, 50, 95]
        q = np.array(q, dtype=np.float64, copy=True)

        perc = np.percentile(self.ensemble, q=q, axis=1,
                             interpolation=interpolation)
        return perc.T


@attr.s()
class UKPrediction(Prediction):
    """MCMC prediction of a UK37 record

    Parameters
    ----------
    ensemble : ndarray
        Ensemble of predictions. A 2d array (nxm) for n predictands and m
        ensemble members.
    prior_mean : float or None, optional
        Prior mean used for the prediction.
    prior_std : float or None, optional
        Prior sample standard deviation used for the prediction.
    """


@attr.s()
class SSTPrediction(Prediction):
    """MCMC prediction of a SST record

    Parameters
    ----------
    ensemble : ndarray
        Ensemble of predictions. A 2d array (nxm) for n predictands and m
        ensemble members.
    prior_mean : float or None, optional
        Prior mean used for the prediction.
    prior_std : float or None, optional
        Prior sample standard deviation used for the prediction.
    jump_distance : float
        Standard deviation of the jump distribution.
    acceptance : float
        Acceptance rate of Metropolis-Hastings MCMC.
    rhat : float
        Median rhat for MCMC convergence.

    References
    ----------
    .. [1] Gelman, Andrew, ed. Bayesian Data Analysis. 2nd ed. Texts in
        Statistical Science. Boca Raton, Fla: Chapman & Hall/CRC, 2004.
    """
    jump_distance = attr.ib(default=None)
    acceptance = attr.ib(default=None)
    rhat = attr.ib(default=None)


def predict_uk(sst):
    """Predict a UK'37 value given SST

    Parameters
    ----------
    sst : 1d array_like
        SST values. Array length is N.

    Returns
    -------
    output : UKPrediction
        Inferred ensemble UK'37 values.
    """

    draws = get_draws()
    b_draws_final = draws['b_draws_final']
    tau2_draws_final = draws['tau2_draws_final']
    knots = draws['knots'].ravel()

    xnew = np.array(sst)

    degree = 2
    aknt = augknt(knots, degree)

    ynew = np.empty((len(xnew), len(b_draws_final)))
    for i, b_now in enumerate(b_draws_final):
        tau2_now = tau2_draws_final[i]

        tck = [aknt, b_now, degree]
        bs = extrapolate_spline(tck)
        mean_now = bs(xnew)

        ynew[:, i] = np.random.normal(mean_now, np.sqrt(tau2_now))

    output = UKPrediction(ensemble=ynew)
    return output


def normpdf(x, mu, sigma):
    """Get PDF for normal distribution at x

    This is faster than scipy.stats.norm.pdf().
    """
    u = (x - mu) / np.abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-u * u / 2)
    return y


def predict_sst(uk, pstd, progressbar=True):
    """Predict SST value given UK'37

    Parameters
    ----------
    uk : 1d array_like
        UK'37 values. Array length is N.
    pstd : float
        Prior standard deviation. Recommended values are 7.5 - 10 for most
        UK'37 data. Lower values are usually fine for UK'37 data with a
        smaller range.
    progressbar : bool, optional
        Whether or not to display a progress bar on the command line. The bar
        shows how many MCMC iterations have been completed.


    Returns
    -------
    output : SSTPrediction
        prior_mean : float
            Prior mean value, taken from the mean of the UK'37 data converted
            to SST with the Prahl equation.
        prior_std : float
            Prior standard deviation (set by user).
        jump_dist : float
            Standard deviation of the jump distribution. Values are chosen
            to achieve an acceptance rate of ca. 0.44 [1]_.

    References
    ----------
    .. [1] Gelman, Andrew, ed. Bayesian Data Analysis. 2nd ed. Texts in
        Statistical Science. Boca Raton, Fla: Chapman & Hall/CRC, 2004.
    """

    # TODO: Add limit to uk range -- I can make strange numbers with large uk vals (e.g. 28)
    draws = get_draws()
    b_draws_final = draws['b_draws_final'][::3, :]
    tau2_draws_final = draws['tau2_draws_final'][::3, :]
    knots = draws['knots'].ravel()

    uk = np.array(uk)

    n_uk = len(uk)
    n_posterior = len(tau2_draws_final)

    # Nsamps
    n_iter = 500
    burnin = 250

    # Set priors. Use prahl conversion to target mean and std
    prior_mean = (uk - 0.039) / 0.034

    # Save priors to output
    pmean_out = prior_mean
    pstd_out = pstd

    # Vectorize priors
    prior_mean = pmean_out * np.ones(n_uk)
    prior_var = pstd_out ** 2 * np.ones(n_uk)

    # Set an initial SST value
    initial_sst = prior_mean

    mh_samples = np.empty((n_uk, n_posterior, n_iter - burnin))
    mh_samples[:] = np.nan
    accepts_t = np.empty((n_uk, n_posterior, n_iter - burnin))
    accepts_t[:] = np.nan

    # Make a spline with set knots
    degree = 2  # order is 3
    kn = augknt(knots, degree)

    prior_mean_median = np.median(prior_mean)
    if prior_mean_median < 20:
        jump_dist = 3.5
    elif 20 <= prior_mean_median <= 23.7:
        jump_dist = 3.7
    else:
        jump_dist = prior_mean_median * 0.8092 - 15.1405

    indices = range(n_posterior)
    if progressbar:
        indices = tqdm(indices, total=n_posterior)

    # MH loop
    for jj in indices:

        accepts = np.empty((n_uk, n_iter))
        accepts[:] = np.nan
        samples = np.empty((n_uk, n_iter))
        samples[:] = np.nan

        # Initialize at starting value
        samples[:, 0] = initial_sst
        sample_now = initial_sst

        b_now = b_draws_final[jj, :]
        tau_now = tau2_draws_final[jj]
        # use spmak to put together the bspline
        tck = [kn, b_now, degree]
        bs = extrapolate_spline(tck)

        # evaluate mean UK value at current SST
        mean_now = bs(sample_now)

        # Evaluate likelihood
        likelihood_now = normpdf(uk, mean_now, np.sqrt(tau_now))

        # Evaluate prior
        prior_now = normpdf(sample_now, prior_mean, np.sqrt(prior_var))

        # multiply to get initial proposal S0
        initial_proposal = likelihood_now * prior_now

        for kk in range(1, n_iter):
            # generate proposal using normal jumping distr.
            proposal = np.random.normal(sample_now, jump_dist)
            # evaluate mean value at current sst
            mean_now = bs(proposal)
            # evaluate liklihood
            likelihood_now = normpdf(uk, mean_now, np.sqrt(tau_now))

            # evaluate prior
            prior_now = normpdf(proposal, prior_mean, np.sqrt(prior_var))
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

    # reshape
    mh_c = mh_samples.reshape([n_uk, n_posterior * (n_iter-burnin)], order='F')

    output = SSTPrediction(ensemble=mh_c,
                           acceptance=np.nansum(accepts_t) / (n_uk * n_posterior * (n_iter - burnin)),
                           rhat=np.median(rhats, axis=0),
                           jump_distance=jump_dist,
                           prior_mean=pmean_out,
                           prior_std=pstd_out)
    return output
