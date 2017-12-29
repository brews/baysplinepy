import os.path
from scipy.io import loadmat


HERE = os.path.abspath(os.path.dirname(__file__))
DRAWS_PATH = os.path.join(HERE, 'bayes_posterior.mat')

draws = loadmat(DRAWS_PATH)