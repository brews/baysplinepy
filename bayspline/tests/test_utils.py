import pytest
import numpy as np
import bayspline.utils as utils


def test_augknt():
    goal = np.array([1, 1, 1, 2, 2, 2])
    victim = utils.augknt([1, 2], 2)
    np.testing.assert_equal(goal, victim)
