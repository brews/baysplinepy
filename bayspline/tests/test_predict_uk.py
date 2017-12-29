import pytest
import numpy as np
import bayspline


def test_predict_uk():
    bayspline.predict_uk(1, 2, 3)
