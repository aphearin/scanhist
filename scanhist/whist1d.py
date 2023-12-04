"""This module calculates triweighted 1d histograms in a variety of ways through
different sequences of jax.vmap, jax.lax.scan, and jax.lax.fori_loop

"""
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from .tw_utils import _tw_bin_jax_kern

# vmap over first and second pairs of arguments in all possible permutations
_singlept_multibin_hist1d_vmap = jjit(
    vmap(_tw_bin_jax_kern, in_axes=[None, None, 0, 0])
)
_multipt_singlebin_hist1d_vmap = jjit(
    vmap(_tw_bin_jax_kern, in_axes=[0, 0, None, None])
)
_multipt_multibin_hist1d_vmap = jjit(
    vmap(_singlept_multibin_hist1d_vmap, in_axes=[0, 0, None, None])
)
_multipt_multibin_hist1d_vmap2 = jjit(
    vmap(_multipt_singlebin_hist1d_vmap, in_axes=[None, None, 0, 0])
)


@jjit
def hist1d_vmap_bins_then_vmap_pts(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over bins, outer loop vmapped over points.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    vmap appends a new dimension to axis=0.
    The outer loop is over points so we sum over axis=0

    """
    X = _multipt_multibin_hist1d_vmap(xarr, dxarr, xloarr, xhiarr)
    return jnp.sum(X, axis=0)


@jjit
def hist1d_vmap_pts_then_vmap_bins(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over points, outer loop vmapped over bins.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    vmap appends a new dimension to axis=0.
    The outer loop is over bins so we sum over the points in axis=1

    """
    X = _multipt_multibin_hist1d_vmap2(xarr, dxarr, xloarr, xhiarr)
    return jnp.sum(X, axis=1)


@jjit
def _singlept_multibin_hist1d_scan(x, dx, xloarr, xhiarr):
    """Scan over an array of bins"""

    @jjit
    def _scan_func(carryover, arr_element):
        xlo, xhi = arr_element
        w = _tw_bin_jax_kern(x, dx, xlo, xhi)
        accumulated = w
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = jnp.array((xloarr, xhiarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return scan_stack


@jjit
def _multipt_singlebin_hist1d_scan(xarr, dxarr, xlo, xhi):
    """Scan over an array of data points"""

    @jjit
    def _scan_func(carryover, arr_element):
        x, dx = arr_element
        w = _tw_bin_jax_kern(x, dx, xlo, xhi)
        accumulated = w + carryover
        carryover = accumulated
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = jnp.array((xarr, dxarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return final_carryover


# vmap over data points after first scanning over bins
# this will return a matrix of shape (npts, nbins) whose first axis must be summed
_singlept_multibin_hist1d_scan_vmap = jjit(
    vmap(_singlept_multibin_hist1d_scan, in_axes=(0, 0, None, None))
)

# vmap over bins after first scanning over points
# this directly returns result with shape (nbins, )
_multipt_singlebin_hist1d_scan_vmap = jjit(
    vmap(_multipt_singlebin_hist1d_scan, in_axes=(None, None, 0, 0))
)


@jjit
def hist1d_scan_bins_then_vmap_pts(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram
    Inner loop scanned over bins, outer loop vmapped over points.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    """
    X = _singlept_multibin_hist1d_scan_vmap(xarr, dxarr, xloarr, xhiarr)
    return jnp.sum(X, axis=0)


@jjit
def hist1d_scan_pts_then_vmap_bins(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram
    Inner loop scanned over points, outer loop vmapped over bins.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    """
    return _multipt_singlebin_hist1d_scan_vmap(xarr, dxarr, xloarr, xhiarr)


@jjit
def hist1d_scan_bins_then_scan_pts(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop scanned over bins, outer loop scanned over data points.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each data point x, the _singlept_multibin_hist1d_scan kernel is called
    to compute x_hist, the nbins-dimensional histogram for the data point.
    That kernel is called npts times within a scan over the data points.

    """

    @jjit
    def _scan_func(carryover, arr_element):
        x, dx = arr_element
        x_hist = _singlept_multibin_hist1d_scan(x, dx, xloarr, xhiarr)
        accumulated = carryover + x_hist
        carryover = accumulated
        return carryover, accumulated

    scan_init = jnp.zeros_like(xloarr)
    scan_arr = jnp.array((xarr, dxarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return final_carryover


@jjit
def hist1d_scan_pts_then_scan_bins(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop scanned over points, outer loop scanned over bins.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each bin, the _multipt_singlebin_hist1d_scan kernel is called
    to compute bin_counts, the total contribution to the bin from all the data.
    That kernel is called nbins times within a scan over the bin boundaries.

    """

    @jjit
    def _scan_func(carryover, arr_element):
        xlo, xhi = arr_element
        bin_counts = _multipt_singlebin_hist1d_scan(xarr, dxarr, xlo, xhi)
        accumulated = bin_counts
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = jnp.array((xloarr, xhiarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return scan_stack


@jjit
def hist1d_vmap_pts_then_scan_bins(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over points, outer loop scanned over bins.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each bin, the _multipt_singlebin_hist1d_vmap kernel is called
    to compute an array of shape (npts, ). This array is then summed to compute
    the total contribution to the bin from all the data.
    That kernel is called nbins times within a vmap over the bin boundaries.

    """

    @jjit
    def _scan_func(carryover, arr_element):
        xlo, xhi = arr_element
        bin_counts = jnp.sum(_multipt_singlebin_hist1d_vmap(xarr, dxarr, xlo, xhi))
        accumulated = bin_counts
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = jnp.array((xloarr, xhiarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return scan_stack


@jjit
def hist1d_vmap_bins_then_scan_pts(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over bins, outer loop scanned over data points.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each data point x, the _singlept_multibin_hist1d_vmap kernel is called
    to compute x_hist, the nbins-dimensional histogram for the data point.
    That kernel is called npts times within a scan over the data points.

    """

    @jjit
    def _scan_func(carryover, arr_element):
        x, dx = arr_element
        x_hist = _singlept_multibin_hist1d_vmap(x, dx, xloarr, xhiarr)
        accumulated = carryover + x_hist
        carryover = accumulated
        return carryover, accumulated

    scan_init = jnp.zeros_like(xloarr)
    scan_arr = jnp.array((xarr, dxarr)).T
    res = lax.scan(_scan_func, scan_init, scan_arr)
    final_carryover, scan_stack = res
    return final_carryover


_tw_bin_jax_kern_vmap_pts = jjit(vmap(_tw_bin_jax_kern, in_axes=(0, 0, None, None)))


@jjit
def hist1d_vmap_pts_then_loop_bins(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over points, outer loop fori_loop over bins.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each bin, the _tw_bin_jax_kern_vmap_pts kernel is called
    to compute an array of weights contributing to the bin. This array is summed
    to compute the total contribution of the data to the bin.
    That kernel is called nbins times within a fori_loop over bins.

    """

    @jjit
    def _loop_func(ibin, args):
        result = args
        xlo = xloarr[ibin]
        xhi = xhiarr[ibin]

        weight_ibin = jnp.sum(_tw_bin_jax_kern_vmap_pts(xarr, dxarr, xlo, xhi))
        result = result.at[ibin].add(weight_ibin)

        return result

    n_bins = xloarr.shape[0]
    lower, upper = 0, n_bins
    init_val = jnp.zeros(n_bins)
    whist = lax.fori_loop(lower, upper, _loop_func, init_val)

    return whist


@jjit
def hist1d_vmap_bins_then_loop_pts(xarr, dxarr, xloarr, xhiarr):
    """Calculate 1-d weighted histogram.
    Inner loop vmapped over bins, outer loop fori_loop over points.

    Parameters
    ----------
    xarr : array-like, shape (npts, )
        Array of input data

    dxarr : array-like, shape (npts, )
        Array of kernel width for each point in input data

    xloarr : array-like, shape (nbins, )
        Array of lower bound on each xbin

    xhiarr : array-like, shape (nbins, )
        Array of upper bound on each xbin

    Returns
    -------
    whist : array-like, shape (nbins, )
        Weighted histogram of input data

    Notes
    -----
    For each point, the _singlept_multibin_hist1d_vmap kernel is called
    to compute x_hist, the nbins-dimensional histogram for the data point.
    That kernel is called npts times within a fori_loop over the data points.

    """

    @jjit
    def _loop_func(idata, args):
        result = args
        x = xarr[idata]
        dx = dxarr[idata]

        x_hist = _singlept_multibin_hist1d_vmap(x, dx, xloarr, xhiarr)
        result = result + x_hist

        return result

    n_pts = xarr.shape[0]
    n_bins = xloarr.shape[0]
    lower, upper = 0, n_pts
    init_val = jnp.zeros(n_bins)
    whist = lax.fori_loop(lower, upper, _loop_func, init_val)

    return whist
