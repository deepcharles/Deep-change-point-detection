import jax.numpy as jnp
from jax import jit
from jax.lax import stop_gradient

from .inference import get_switching_model


@jit
def get_mse_loss(signal: jnp.ndarray, centroids: jnp.ndarray, penalty: float) -> float:
    # signal, shape (n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    # penalty, float
    optimal_state_sequence = stop_gradient(
        get_switching_model(signal=signal, centroids=centroids, penalty=penalty)
    )
    approx = centroids[optimal_state_sequence]
    n_samples, n_dims = signal.shape
    mse = jnp.square(signal - approx).sum() / n_samples
    return mse
