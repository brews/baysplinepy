import pytest
import numpy as np
import scipy.interpolate as interpolate
import bayspline.utils as utils


@pytest.fixture(scope='session')
def ppolyspline1():
    x = np.array([-0.4,  23.6,  29.6])
    c = np.array([[4.60901815e-05, -2.54546446e-03],
                  [3.22603382e-02, 3.44726669e-02],
                  [5.32351561e-02, 8.54031217e-01]])
    return interpolate.PPoly(x=x, c=c)


@pytest.fixture(scope='session')
def tck1():
    # Knots, coefficients, polynomial degree:
    tck = [np.array([-0.4, -0.4, -0.4, 23.6, 29.6, 29.6, 29.6]),
           np.array([0.05323516, 0.44035921, 0.95744922, 0.9692305]),
           2]
    return tck


def test_augknt():
    """Test basic utils.augknt()
    """
    goal = np.array([1, 1, 1, 2, 2, 2])
    victim = utils.augknt([1, 2], 2)
    np.testing.assert_equal(goal, victim)


def test_bspline2ppoly(tck1):
    """Test basic utils.bspline2poly()
    """
    goal_x = np.array([-0.4,  23.6,  29.6])
    goal_c = np.array([[4.60901815e-05, -2.54546446e-03],
                       [3.22603382e-02, 3.44726669e-02],
                       [5.32351561e-02, 8.54031217e-01]])
    victim = utils.bspline2ppoly(tck1)
    np.testing.assert_allclose(victim.x, goal_x, atol=1e-4)
    np.testing.assert_allclose(victim.c, goal_c, atol=1e-4)


def test_add_ppolyknot1(ppolyspline1):
    goal_x = np.array([-1.4, -0.4, 23.6, 29.6, 30.6])
    goal_c = np.array([[4.60901815e-05, 4.60901815e-05,
                        -2.54546446e-03, -2.54546446e-03],
                       [3.21681578e-02, 3.22603382e-02,
                        3.44726669e-02, 3.92709337e-03],
                       [2.10209081e-02, 5.32351561e-02,
                        8.54031217e-01, 9.69230498e-01]])
    victim = utils.add_ppolyknot(ppolyspline1, knots=[-1.4,  30.6])
    np.testing.assert_allclose(victim.x, goal_x, atol=1e-3)
    np.testing.assert_allclose(victim.c, goal_c, atol=1e-3)


def test_add_ppolyknot2():
    goal = interpolate.PPoly(x=np.array([-1.4, -0.4, 23.6]),
                              c=np.array([[0.0323, 0.0323], [0.0210, 0.0532]]))
    test = interpolate.PPoly(x=np.array([-0.4, 23.6]),
                              c=np.array([[0.0323], [0.0532]]))
    victim = utils.add_ppolyknot(test, knots=[-1.4])
    np.testing.assert_allclose(victim.x, goal.x, atol=1e-4)
    np.testing.assert_allclose(victim.c, goal.c, atol=1e-4)


def test_add_ppolyknot3(ppolyspline1):
    goal_x = np.array([-3.4, -0.4, 23.6, 28.6, 29.6])
    goal_c = np.array([[0.0000, 0.0000, -0.0025, -0.0025],
                       [0.0320, 0.0323, 0.0345, 0.0090],
                       [-0.0431, 0.0532, 0.8540, 0.9628]])
    victim = utils.add_ppolyknot(ppolyspline1, knots=[-3.4, 28.6])
    np.testing.assert_allclose(victim.x, goal_x, atol=1e-3)
    np.testing.assert_allclose(victim.c, goal_c, atol=1e-3)


def test_extrapolate_spline(tck1):
    victim = utils.extrapolate_spline(tck1, degree=1)
    goal_c = np.array([[0, 0.0323, 0.0210],
                       [0.0000, 0.0323, 0.0532],
                       [-0.0025, 0.0345, 0.8540],
                       [0, 0.0039, 0.9692]])
    goal_x = np.array([-1.4000, -0.4000, 23.6000, 29.6000, 30.6000])
    np.testing.assert_allclose(victim.x, goal_x, atol=1e-3)
    np.testing.assert_allclose(victim.c, goal_c.T, atol=1e-3)
