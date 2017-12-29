import pytest
import numpy as np
import bayspline


def test_predict_sst():
    np.random.seed(123)

    uk = np.array([0.4, 0.5, 0.6])
    age = np.array([1, 2, 3])
    pstd = 7.5

    # Got these goals from original MATLAB code output.
    goal_prior_mean = 13.5588
    goal_prior_std = 7.5
    goal_jump_dist = 3.5
    goal_rhat = 1.0074
    goal_accepts = 0.4318
    goal_sst = np.array([[7.9742, 8.8912, 10.2884, 11.6962, 12.6191],
                         [10.8206, 11.7538, 13.1597, 14.5701, 15.5144],
                         [13.7086, 14.6448, 16.0522, 17.4940, 18.3874]])

    victim = bayspline.predict_sst(age, uk, pstd)

    np.testing.assert_allclose(goal_prior_mean, victim['prior_mean'], atol=1e-4)
    assert goal_prior_std == victim['prior_std']
    assert goal_jump_dist == victim['jump_dist']
    np.testing.assert_allclose(goal_rhat, victim['rhat'], atol=5e-2)
    np.testing.assert_allclose(goal_accepts, victim['accepts'], atol=0.5)
    np.testing.assert_allclose(goal_sst, victim['sst'], atol=1)
