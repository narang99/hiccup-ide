import jax
from flax import nnx
import optax
from pt_to_api.benchmark.init_strats import (
    InitStrategy,
    StandardInitStrategy,
    NoInitStrategy,
)
from pt_to_api.benchmark.jax.core import Autoencoder


@nnx.vmap(in_axes=(0, None, None, None, None, None), out_axes=0)
def parallel_init_models(
    seeds, input_dim, n_components, scaled_hyperparameters, init_strategy, lr
):
    return init_single_model(
        seeds, input_dim, n_components, scaled_hyperparameters, init_strategy, lr
    )


def init_single_model(
    s, input_dim, n_components, scaled_hyperparameters, init_strategy, lr
):
    jax.debug.print("init model seed {seed_val}", seed_val=s)
    rngs = nnx.Rngs(s)
    model = Autoencoder(input_dim, n_components, rngs)
    # Initialize model parameters using shared hyperparameters
    init_model_parameters(
        model, scaled_hyperparameters, n_components, init_strategy, rngs
    )
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        recon_loss=nnx.metrics.Average("recon_loss"),
        weight_loss=nnx.metrics.Average("weight_loss"),
        codes_loss=nnx.metrics.Average("codes_loss"),
        mse=nnx.metrics.Average("mse"),
    )
    return model, optimizer, metrics


def init_model_parameters(
    model: nnx.Module,
    scaled_hyperparameters: dict,
    n_components: int,
    init_strategy: InitStrategy,
    rngs: nnx.Rngs,
):
    """Initialize JAX/NNX model parameters using proper PRNG keys."""
    if isinstance(init_strategy, StandardInitStrategy):
        # Standard normal initialization for encoder and decoder
        sigma_enc = scaled_hyperparameters["sigma_enc"]
        sigma_0 = scaled_hyperparameters["sigma_0"]

        # Initialize encoder weights with normal distribution
        encoder_key = rngs.params()
        encoder_weights = (
            jax.random.normal(encoder_key, model.encoder.kernel.shape) * sigma_enc
        )
        model.encoder.kernel.value = encoder_weights

        # Initialize decoder weights with normal distribution
        decoder_key = rngs.params()
        decoder_weights = (
            jax.random.normal(decoder_key, model.decoder.kernel.shape) * sigma_0
        )
        model.decoder.kernel.value = decoder_weights

    elif isinstance(init_strategy, NoInitStrategy):
        # Keep default initialization
        print("using default NNX initialization")
    else:
        raise ValueError(f"Unsupported initialization strategy: {init_strategy}")
