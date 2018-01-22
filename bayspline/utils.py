import os
import pkgutil
import io
import numpy as np


def augknt(knots, degree):
    """Augment knots to meet boundary conditiions

    Python version of MATLAB's augknt().
    """
    heads = [knots[0]] * degree
    tails = [knots[-1]] * degree
    return np.concatenate([heads, knots, tails])


def chainconvergence(chains, m):
    """ calculate the R-hat stat
    """
    # TODO(brews): Finish function docstring
    # TODO(brews): Finish function test

    # From Jess Tierney's "ChainConvergence.m"
    # %
    # % function [Rhat, Neff]=ChainConvergence(chains, M)
    # %
    # % calculate the R-hat stat form "Bayesian Data Analysis" (page 297) for
    # % monitoring  convergence of multiple parallel MCMC runs. Also outputs
    # % n_eff from page 298.
    # % chains: a matrix of MCMC chains, each of the same length.
    # % M: the number of different chains - must be one of the dimensions of
    # % chains.
    # %
    # %test
    # %chains=randn(100,50);
    # %chains(:,1)=chains(:,1)+10;
    # %M=5;
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


def get_example_data(filename):
    """Get a BytesIO object for a bayspar example file.

    Parameters
    ----------
    filename : str
        File to load.

    Returns
    -------
    BytesIO of the example file.
    """
    resource_str = os.path.join('example_data', filename)
    return io.BytesIO(pkgutil.get_data('bayspline', resource_str))
