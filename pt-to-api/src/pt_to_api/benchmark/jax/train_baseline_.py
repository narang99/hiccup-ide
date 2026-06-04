from pt_to_api.benchmark.core import SingleRun
import jax.numpy as jnp
from flax import nnx
from pt_to_api.benchmark.jax.core import Autoencoder, get_batches
from pt_to_api.benchmark.jax.steps import baseline_train_step
import numpy as np
import optax


def train_baseline(
    X,
    n_components,
    rngs,
    lr=1e-3,
    epochs=2000,
    batch_size=256,
    verbose=True,
) -> SingleRun:
    """
    Train the autoencoder with just reconstruction loss to find baseline.

    Args:
        X: numpy array (n_samples, input_dim)
        n_components: number of dictionary atoms
        lr: learning rate
        epochs: number of training epochs
        batch_size: batch size for training
        verbose: whether to print progress
        seed: random seed

    Returns:
        SingleRun with baseline model results
    """
    print(f"training JAX baseline model, epochs={epochs}")

    input_dim = X.shape[1]
    X_jax = jnp.array(X, dtype=jnp.float32)
    rngs = nnx.Rngs(default=0, training_loop=1)

    # Initialize model
    model = Autoencoder(input_dim, n_components, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    # Training loop
    key = rngs.training_loop()
    for epoch in range(epochs):
        # Shuffle data
        for key, batch in get_batches(X_jax, batch_size, key):
            baseline_train_step(model, optimizer, metrics, batch)

        if verbose and epoch % 200 == 0:
            loss = metrics.compute()["loss"]
            metrics.reset()
            print(f"baseline epoch {epoch:4d} | recon_loss {float(loss):.4f}")

    # Final evaluation
    recon, codes, _ = model(X_jax)
    final_loss = jnp.mean((X_jax - recon) ** 2)

    # JAX has linear shape: [input, output]
    # encoder: [dimensions, n-components]
    # decoder: [n-components, dimensions]
    # both in single run should be stored as [n-components, dimensions]
    return SingleRun(
        codes=np.array(codes),
        encoder=np.array(model.encoder.kernel.T),
        components=np.array(model.decoder.kernel),
        recon=np.array(recon),
        loss=float(final_loss),
        hyperparameters={},
    )
