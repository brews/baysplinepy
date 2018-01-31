import os.path
from copy import deepcopy
from pkgutil import get_data
from io import BytesIO
from scipy.io import loadmat


def get_matlab_resource(resource, package='bayspline', **kwargs):
    """Read flat MATLAB files as package resources, output for Numpy"""
    with BytesIO(get_data(package, resource)) as fl:
        data = loadmat(fl, **kwargs)
    return data


draws = get_matlab_resource('modelparams/bayes_posterior.mat')


def get_draws():
    """Get model parameter draws for MCMC
    """
    return deepcopy(draws)
