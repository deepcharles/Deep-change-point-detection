import jax.numpy as jnp
from jax import jit
from jax.lax import stop_gradient

from .inference import get_switching_model


@jit
def get_mse_loss(signal: jnp.ndarray, centroids: jnp.ndarray, penalty: float) -> float:
    # signal, shape (n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    # penalty, float
    # The output is the penalized mse. The penalty is n_changes * penalty.
    optimal_state_sequence = stop_gradient(
        get_switching_model(signal=signal, centroids=centroids, penalty=penalty)
    )
    approx = centroids[optimal_state_sequence]
    n_samples, n_dims = signal.shape
    n_changes = jnp.count_nonzero(jnp.diff(optimal_state_sequence))
    mse = (jnp.square(signal - approx).sum() + penalty * n_changes) / n_samples
    return mse
