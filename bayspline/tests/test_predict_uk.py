import pytest
import numpy as np
import bayspline


def test_predict_uk():
    np.random.seed(123)

    sst = np.array([1, 15, 30])
    age = np.array([1, 2, 3])

    goal_uk_mean = np.array([0.09384701, 0.5593857, 0.98108519])

    victim = bayspline.predict_uk(age, sst)
    output = victim['uk'].mean(axis=1)

    np.testing.assert_allclose(goal_uk_mean, output, atol=1e-2)
