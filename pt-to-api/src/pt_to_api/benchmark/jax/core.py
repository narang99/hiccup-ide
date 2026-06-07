from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from pt_to_api.benchmark.core import SingleRun


class ModelTrainStepUnconditionalParams(NamedTuple):
    alpha: jnp.ndarray
    recon_coeff: jnp.ndarray
    codes_coeff: jnp.ndarray
    weights_coeff: jnp.ndarray
    ln_term_coeff: jnp.ndarray


class Autoencoder(nnx.Module):
    def __init__(self, input_dim, n_components, rngs):
        self.encoder = nnx.Linear(input_dim, n_components, rngs=rngs, use_bias=False)
        self.decoder = nnx.Linear(n_components, input_dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        # codes = self.encoder(x)
        # try positive codes only
        codes = self.encoder(x) ** 2
        latent = codes[..., None] * self.decoder.kernel[None, ...]
        recon = jnp.sum(latent, axis=1)
        latent_perm = jnp.transpose(latent, (0, 2, 1))  # [batch, columns, components]
        return recon, codes, latent_perm


def autoencoder_from_single_run(run: SingleRun) -> Autoencoder:
    n_components, input_dim = run.encoder.shape
    model = Autoencoder(input_dim, n_components, nnx.Rngs(0))
    # encoder wants [dimensions, n_components]
    model.encoder.kernel.value = run.encoder.T
    model.decoder.kernel.value = run.components
    # model.encoder.kernel.set_value(run.encoder.T)
    # model.decoder.kernel.set_value(run.components)
    return model


def get_batches(X, batch_size, key=None):
    if key is not None:
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, len(X))
    else:
        idx = jnp.arange(len(X))
    for i in range(0, len(X), batch_size):
        yield key, X[idx[i : i + batch_size]]
