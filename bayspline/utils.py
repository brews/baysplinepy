import os
import pkgutil
import io
import numpy as np
import scipy.interpolate as interpolate
import scipy.special as special


def bspline2ppoly(tck, extrapolate=None):
    """Build a piecewise polynomial from a spline, removing buffer knots

    Parameters
    ----------
    tck
        A spline, as returned by `splrep` or a BSpline object.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs.
        If 'periodic', periodic extrapolation is used. Default is True.

    Returns
    -------
    ppoly: interpolate.PPoly
        A interpolate.PPoly instance.

    """
    if isinstance(tck, interpolate.BSpline):
        t, c, k = tck.tck
        if extrapolate is None:
            extrapolate = tck.extrapolate
    else:
        t, c, k = tck

    t = t[k:-k]  # Trim buffer knots away.

    cvals = np.empty((k + 1, len(t) - 1), dtype=c.dtype)
    for m in range(k, -1, -1):
        y = interpolate.fitpack.splev(t[:-1], tck, der=m)
        cvals[k - m, :] = y / special.gamma(m + 1)

    ppoly = interpolate.PPoly.construct_fast(cvals, t, extrapolate)
    return ppoly


def add_ppolyknot(ppoly, knots):
    """Add knots to univariate piecewise polynomial

    Parameters
    ----------
    ppoly : interpolate.PPoly
        A univariate piecewise polynomial.
    knots : sequence
        Breakpoints to add to polynomial.

    Returns
    -------
    ppoly_new : interpolate.PPoly
        Copy of piecewise polynomial with added knots.
    """
    knots = np.array(knots)
    knots.sort()

    b = ppoly.x.copy()
    k = len(ppoly.c)  # Polynomial order
    c = ppoly.c.T.copy()
    pieces = int(np.ceil(len(b) / 2))  # Number of pieces, I think

    lb = len(knots)

    index0 = np.where(knots < b[0])[0]
    l0 = len(index0)

    index2 = np.where(knots > b[pieces])[0]
    l2 = len(index2)

    index1 = np.arange(l0, (lb - l2))
    if index1.size < 1:
        index = index1
        jl = np.concatenate([np.ones(l0) - 1,
                             np.kron(np.ones(l2), pieces - 1)]).astype('int')
    else:
        # Find `knots` not in `b`, make their left-most point `b[link[index]]`
        link = np.searchsorted(b[:pieces], knots[index1], side='right') - 1
        index = np.where(b[link] != knots[index1])[0]
        jl = np.concatenate([np.ones(l0) - 1,
                             link[index],
                             np.kron(np.ones(l2), pieces - 1)]).astype('int')
    ljl = len(jl)

    # Return input if all of `knots` already in `ppoly`.
    if ljl == 0:
        return ppoly

    if l2 > 0:
        tmp = knots[lb - 1]
        knotsin = np.concatenate([knots[(lb - 2):(lb - l2 - 1):-1], b[[pieces]]])
        knots[(lb - 1):(lb - l2 - 1):-1] = knotsin
        b[pieces] = tmp

    addknots = knots[np.concatenate([index0, index1[index], index2])]
    x = addknots - b[[jl]]
    a = c[[jl]]
    for i in range(k - 1, 0, -1):
        for j in range(1, i + 1):
            a[:, j] = x * a[:, j - 1] + a[:, j]

    newknots = np.concatenate([b, addknots])
    newknots.sort()

    newc = np.zeros((len(newknots) - 1, k))
    newc[np.searchsorted(newknots, b[:pieces], side='right') - 1, :] = c
    newc[np.searchsorted(newknots, addknots, side='right') - 1, :] = a

    ppoly_new = interpolate.PPoly(c=newc.T, x=newknots)
    return ppoly_new


def extrapolate_spline(spline, degree=1):
    """Extrapolate 1D function beyond its basic interval
    """
    ppoly = bspline2ppoly(spline)

    breaks = np.array(ppoly.x)
    coefs = np.array(ppoly.c)
    splinedegree = len(breaks) - 1

    # Add breaks outside of both ends (-1, +1)
    newbreaks = np.array([breaks[0] - 1, breaks[-1] + 1])
    ppoly = add_ppolyknot(ppoly, newbreaks)

    splinedegree = ppoly.c.shape[0] - 1
    if splinedegree <= degree:
        return ppoly

    newcoefs2 = np.array(ppoly.c)
    degdif = splinedegree - degree

    # In first and last polynomial, set terms greater than `degree` to 0.
    newcoefs2[:degdif, -1:] = 0
    newcoefs2[:degdif, 0] = 0

    minipoly = interpolate.PPoly(c=newcoefs2[degdif:splinedegree + 1, 1:2],
                                 x=ppoly.x[1:3])
    c1 = add_ppolyknot(minipoly, ppoly.x[[0]]).c
    newcoefs2[degdif:splinedegree + 1, 0] = c1[:, 0]

    return interpolate.PPoly(c=newcoefs2, x=ppoly.x)


def augknt(knots, degree):
    """Augment knots to meet boundary conditions

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
    """Get a BytesIO object for a bayspline example file.

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
