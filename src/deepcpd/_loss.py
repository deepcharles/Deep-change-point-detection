from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit
from jax.lax import scan, while_loop
from jax.tree_util import Partial

from .utils import fill_diagonal, softmin


def update_soc_vec(carry, new_cost_vec):
    # new_cost_vec, shape (n_states,)
    # state changes at each iteration, confg never changes
    state, config = carry
    (soc_vec,) = state
    (transition_penalty_mat,) = config
    new_soc_vec = jnp.min(transition_penalty_mat + soc_vec, axis=1) + new_cost_vec

    # update carry
    state = (new_soc_vec,)
    carry = state, config

    return carry, new_soc_vec


@jit
def get_switching_model_loss(
    signal: jnp.ndarray, centroids: jnp.ndarray, penalty: float
) -> float:
    # signal, shape (n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    time_axis, state_axis, dim_axis = 0, 1, 2

    # compute all costs
    dist_to_centroids = jnp.square(signal[:, None, :] - centroids[None, ...]).sum(
        axis=dim_axis
    )  # shape (n_samples, n_states)

    # define transition penalty matrix
    n_samples, n_states = dist_to_centroids.shape
    transition_penalty_mat = fill_diagonal(
        jnp.full((n_states, n_states), penalty, dtype=jnp.float32), 0.0
    )

    # carry init
    soc_vec = dist_to_centroids[0]

    state_init = (soc_vec,)
    config = (transition_penalty_mat,)
    carry_init = state_init, config

    # perform the scan
    _, soc_mat = scan(f=update_soc_vec, init=carry_init, xs=dist_to_centroids[1:])
    # soc_mat has shape (n_samples-1, n_states)

    return jnp.min(soc_mat[-1])


def update_soc_vec_soft(carry, new_cost_vec):
    # new_cost_vec, shape (n_states,)
    # state changes at each iteration, confg never changes
    state, config = carry
    (soc_vec,) = state
    (transition_penalty_mat, gamma) = config
    new_soc_vec = (
        softmin(transition_penalty_mat + soc_vec, gamma=gamma, axis=1) + new_cost_vec
    )

    # update carry
    state = (new_soc_vec,)
    carry = state, config

    return carry, new_soc_vec


@jit
def get_switching_model_loss_soft(
    signal: jnp.ndarray, centroids: jnp.ndarray, penalty: float, gamma: float = 1.0
) -> float:
    # signal, shape (n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    time_axis, state_axis, dim_axis = 0, 1, 2

    # compute all costs
    dist_to_centroids = jnp.square(signal[:, None, :] - centroids[None, ...]).sum(
        axis=dim_axis
    )  # shape (n_samples, n_states)

    # define transition penalty matrix
    n_samples, n_states = dist_to_centroids.shape
    transition_penalty_mat = fill_diagonal(
        jnp.full((n_states, n_states), penalty, dtype=jnp.float32), 0.0
    )

    # carry init
    soc_vec = dist_to_centroids[0]

    state_init = (soc_vec,)
    config = (transition_penalty_mat, gamma)
    carry_init = state_init, config

    # perform the scan
    _, soc_mat = scan(f=update_soc_vec_soft, init=carry_init, xs=dist_to_centroids[1:])
    # soc_mat has shape (n_samples-1, n_states)

    return softmin(soc_mat[-1], gamma=gamma)
