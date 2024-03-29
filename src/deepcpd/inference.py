from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax.lax import scan, while_loop
from jax.tree_util import Partial

from .utils import fill_diagonal


@jit
def update_soc_and_state_vec(carry, new_cost_vec: jnp.ndarray):
    # new_cost_vec, shape (n_states,)
    # state changes at each iteration, confg never changes
    state, config = carry
    (soc_vec,) = state
    (transition_penalty_mat,) = config
    new_soc_vec = (transition_penalty_mat + soc_vec).min(axis=1) + new_cost_vec
    new_state_vec = (transition_penalty_mat + soc_vec).argmin(axis=1)

    # update carry
    state = (new_soc_vec,)
    carry = state, config

    return carry, (new_soc_vec, new_state_vec)


@jit
def update_state_sequence(carry: Tuple[jnp.ndarray, jnp.ndarray, int, int]):
    # for the while loop of the backtracking
    state_mat, state_vec, state, end = carry
    # state_mat, shape (n_samples-1, n_states)
    # state_vec, shape (n_samples,)
    # state, int
    # end, int
    state_vec = state_vec.at[end - 1].set(state)
    state = state_mat[end - 2, state]
    # update carry
    carry = state_mat, state_vec, state, end - 1
    return carry


@jit
def continue_backtracking(carry: Tuple[jnp.ndarray, jnp.ndarray, int, int]) -> bool:
    # for the while loop of the backtracking
    state_mat, state_vec, state, end = carry
    # state_mat, shape (n_samples-1, n_states)
    # state_vec, shape (n_samples,)
    # state, int
    # end, int
    return jnp.logical_and(state > -1, end > 0)


@jit
def get_switching_model(
    signal: jnp.ndarray, centroids: jnp.ndarray, penalty: float
) -> jnp.ndarray:
    # signal, shape (n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    # output the optimal sequence of states, shape (n_samples,)
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

    # perform the forward pass
    _, (soc_mat, state_mat) = scan(
        f=update_soc_and_state_vec, init=carry_init, xs=dist_to_centroids[1:]
    )
    # soc_mat and state_mat have shape (n_samples-1, n_states)

    # Backtracking, to find the optimal state sequence
    state_init = jnp.argmin(soc_mat[-1])
    end = n_samples
    state_vec_init = jnp.empty(n_samples, dtype=jnp.int32)
    carry_init = state_mat, state_vec_init, state_init, n_samples
    _, state_vec, _, _ = while_loop(
        cond_fun=continue_backtracking,
        body_fun=update_state_sequence,
        init_val=carry_init,
    )

    return state_vec
