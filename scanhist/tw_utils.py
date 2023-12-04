"""Triweight kernels in JAX
"""
from jax import jit as jjit
from jax import lax


@jjit
def _tw_jax_kern(x, m, h):
    """Triweight kernel

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel

    m : array-like or scalar
        The mean of the kernel

    h : array-like or scalar
        The approximate 1-sigma width of the kernel

    Returns
    -------
    kern : array-like or scalar
        The value of the kernel

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 0.0,
            lambda xx: 35 / 96 * (1 - (xx / 3) ** 2) ** 3 / h,
            x,
        ),
        y,
    )


@jjit
def _tw_cuml_jax_kern(x, m, h):
    """CDF of the triweight kernel

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel

    m : array-like or scalar
        The mean of the kernel

    h : array-like or scalar
        The approximate 1-sigma width of the kernel

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 1.0,
            lambda xx: (
                -5 * xx**7 / 69984
                + 7 * xx**5 / 2592
                - 35 * xx**3 / 864
                + 35 * xx / 96
                + 1 / 2
            ),
            x,
        ),
        y,
    )


@jjit
def _tw_bin_jax_kern(m, h, L, H):
    """Integrated bin weight for the triweight kernel

    Parameters
    ----------
    m : array-like or scalar
        The value at which to evaluate the kernel

    h : array-like or scalar
        The approximate 1-sigma width of the kernel

    L : array-like or scalar
        The lower bin limit

    H : array-like or scalar
        The upper bin limit

    Returns
    -------
    bin_prob : array-like or scalar
        The value of the kernel integrated over the bin

    """
    return _tw_cuml_jax_kern(H, m, h) - _tw_cuml_jax_kern(L, m, h)
