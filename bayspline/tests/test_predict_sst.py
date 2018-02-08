import pytest
import numpy as np
import bayspline as bsl


def test_predict_sst():
    np.random.seed(123)

    uk = np.array([0.4, 0.5, 0.6])
    pstd = 7.5

    # Got these goals from original MATLAB code output.
    goal_prior_mean = np.array([10.6176, 13.5588, 16.5000])
    goal_prior_std = 7.5
    goal_jump_dist = 3.5
    goal_rhat = 1.0074
    goal_accepts = 0.4318
    goal_sst = np.array([[7.9742, 8.8912, 10.2884, 11.6962, 12.6191],
                         [10.8206, 11.7538, 13.1597, 14.5701, 15.5144],
                         [13.7086, 14.6448, 16.0522, 17.4940, 18.3874]])

    victim = bsl.predict_sst(uk=uk, pstd=pstd)

    np.testing.assert_allclose(goal_prior_mean, victim.prior_mean, atol=1e-4)
    assert goal_prior_std == victim.prior_std
    assert goal_jump_dist == victim.jump_distance
    np.testing.assert_allclose(goal_rhat, victim.rhat, atol=5e-2)
    np.testing.assert_allclose(goal_accepts, victim.acceptance, atol=0.5)
    np.testing.assert_allclose(goal_sst, victim.percentile(q=[5, 16, 50, 84, 95]), atol=1)


def test_normpdf():
    victim = bsl.predict.normpdf(np.array([1.5, 2, 3]), 3, 2)
    goal = np.array([0.15056872, 0.17603266, 0.19947114])
    np.testing.assert_allclose(victim, goal, atol=1e-4)


@pytest.mark.skip(reason='Need a goal for test')
def test_predict_sst_real():
    """Test against real data we have as an example
    """
    np.random.seed(123)

    d = np.genfromtxt(bsl.get_example_data('tierney2016-p178-15p.csv'),
                      delimiter=',', names=True)

    prediction = bsl.predict_sst(age=d['age'], uk=d['uk37'], pstd=10)
