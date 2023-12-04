"""
"""
import numpy as np
from jax import random as jran

from .. import whist1d

SEED = 43
TOL = 1e-4

XBIN_BOUNDARIES = np.array(
    [(-5.0, -4.0), (-2.0, -1.0), (0.0, 4.0), (2.0, 5.0), (5.0, 8.0)]
)
XLOARR, XHIARR = XBIN_BOUNDARIES[:, 0], XBIN_BOUNDARIES[:, 1]


def test_multipt_multibin_hist1d():
    """Enforce agreement between different versions of the kernel"""
    npts = 500
    n_tests = 100

    ran_key = jran.PRNGKey(SEED)
    for __ in range(n_tests):
        x_key, dx_key, ran_key = jran.split(ran_key, 3)
        xarr = jran.uniform(x_key, minval=-10, maxval=10, shape=(npts,))
        dxarr = jran.uniform(dx_key, minval=0, maxval=10, shape=(npts,))

        args = (xarr, dxarr, XLOARR, XHIARR)

        result0 = whist1d.hist1d_vmap_bins_then_vmap_pts(*args)

        result1 = whist1d.hist1d_scan_bins_then_scan_pts(*args)
        assert np.allclose(result0, result1, atol=TOL)

        result2 = whist1d.hist1d_vmap_pts_then_vmap_bins(*args)
        assert np.allclose(result0, result2, atol=TOL)

        result3 = whist1d.hist1d_scan_pts_then_scan_bins(*args)
        assert np.allclose(result0, result3, atol=TOL)

        result4 = whist1d.hist1d_scan_bins_then_vmap_pts(*args)
        assert np.allclose(result0, result4, atol=TOL)

        result5 = whist1d.hist1d_scan_pts_then_vmap_bins(*args)
        assert np.allclose(result0, result5, atol=TOL)

        result6 = whist1d.hist1d_vmap_pts_then_scan_bins(*args)
        assert np.allclose(result0, result6, atol=TOL)

        result7 = whist1d.hist1d_vmap_bins_then_scan_pts(*args)
        assert np.allclose(result0, result7, atol=TOL)

        result8 = whist1d.hist1d_vmap_pts_then_loop_bins(*args)
        assert np.allclose(result0, result8, atol=TOL)

        result9 = whist1d.hist1d_vmap_bins_then_loop_pts(*args)
        assert np.allclose(result0, result9, atol=TOL)
