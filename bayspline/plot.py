import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


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
