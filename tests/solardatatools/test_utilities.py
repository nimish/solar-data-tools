import numpy as np

# from solardatatools.signal_decompositions import l1_l2d2p365
from solardatatools.utilities import basic_outlier_filter


def test_basic_outlier_filter():
    np.random.seed(42)
    x = np.random.normal(size=20)
    x[0] *= 5
    msk = basic_outlier_filter(x)
    assert np.sum(~msk) == 1
    np.testing.assert_almost_equal(x[~msk][0], 2.4835707650561636)
