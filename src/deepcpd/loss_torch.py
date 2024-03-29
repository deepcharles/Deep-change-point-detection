# https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9

import jax
import jax.numpy as jnp
import torch
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

from .inference import get_optimal_state_sequence_batch


def j2t(x_jax: jnp.ndarray) -> torch.Tensor:
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch


def t2j(x_torch: torch.Tensor) -> jnp.ndarray:
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    shape = x_torch.shape
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch.flatten()))
    return x_jax.reshape(shape)


class SwitchingModel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch, centroids, penalty):
        batch_jax = t2j(batch)  # shape (n_signals, n_samples, n_dims)
        centroids_jax = t2j(centroids)
        optimal_state_sequence_jax = get_optimal_state_sequence_batch(
            batch=batch_jax, centroids=centroids_jax, penalty=penalty
        )
        optimal_state_sequence_torch = j2t(optimal_state_sequence_jax)
        return optimal_state_sequence_torch


def get_loss_torch(batch: torch.Tensor, centroids: torch.Tensor, penalty: float):
    # batch, shape (n_signals, n_samples, n_dims)
    # centroids, shape (n_states, n_dims)
    # penalty, float
    # compute the penalized loss
    n_signals, n_samples, n_dims = batch.shape
    batch_of_optimal_sequences = SwitchingModel.apply(batch, centroids, penalty)
    n_changes = torch.count_nonzero(
        torch.diff(batch_of_optimal_sequences, axis=1), axis=1
    ).sum()
    batch_approx = centroids[batch_of_optimal_sequences]
    residual_squared_norm = torch.sum((batch - batch_approx) ** 2) / n_samples
    loss_val = (residual_squared_norm + penalty * n_changes) / n_signals
    return loss_val
