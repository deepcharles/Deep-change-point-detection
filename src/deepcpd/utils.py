import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
from jax.tree_util import Partial


@jit
def fill_diagonal(a: jnp.ndarray, val: float) -> jnp.ndarray:
    """Fills the diagonal of a 2D or higher-dimensional array with a specified value.

    Args:
        a (jax.numpy.ndarray): Input array. Must have at least 2 dimensions.
        val (float): Value to fill the diagonal with.

    Returns:
        jax.numpy.ndarray: Array with diagonal filled with the specified value.

    Raises:
        AssertionError: If the input array has fewer than 2 dimensions.

    Example:
        >>> a = jnp.zeros((3, 3))
        >>> filled = fill_diagonal(a, 5)
        >>> filled
        DeviceArray([[5., 0., 0.],
                     [0., 5., 0.],
                     [0., 0., 5.]], dtype=float32)
    """
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


@Partial(jit, static_argnames=("axis",))
def softmin(arr: jnp.ndarray, gamma: float = 1.0, axis=None):
    """Compute the softmin of an array along a specified axis.

    The softmin is a smooth approximation to the minimum function, obtained by
    exponentiating and normalizing the negative of the input array.

    Args:
        arr (jax.numpy.ndarray): Input array.
        gamma (float): Scaling factor. Defaults to 1.0.
        axis (int or tuple of ints): Axis or axes along which to compute the softmin.

    Returns:
        jax.numpy.ndarray: Softmin of the input array along the specified axis.

    Example:
        >>> arr = jnp.array([1.0, 2.0, 3.0])
        >>> softmin(arr)
        Array(0.592394, dtype=float32)
    """
    return -logsumexp(-arr * gamma, axis=axis) / gamma
