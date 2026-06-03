"""
JAX/NNX implementation of autoencoder training with sparse dictionary learning.
"""
import time
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import optax
import numpy as np
from dataclasses import dataclass
from typing import Literal
import math

from .utils import SingleRun, InitStrategy, StandardInitStrategy
from .train_x import (
    cosine_anneal, CosineAnnealReconError, ConstantReconError, 
    AdamOptimType, SGDOptimType, NoSchedType, CosineAnnealingWithWarmRestartsSchedType
)
from .train import (
    recon_loss, gauss_loss, get_hyperparameters_and_init, get_scaled_hyperparamters,
    get_inferred_sigma_eps_if_needed
)


def weights_loss_batched(alpha, sigma_0, W_batch, start_idx=0):
    """Batched version without cyclic shifts. W_batch: [B, C, K]"""
    W_rolled = jnp.roll(W_batch, -start_idx, axis=2)
    W_sq = W_rolled ** 2
    cumsum = jnp.cumsum(W_sq, axis=2)
    phi = alpha * jnp.roll(cumsum, 1, axis=2) + 1
    phi = phi.at[:, :, 0].set(1)
    comp1 = (W_sq * phi / (sigma_0 * sigma_0)).sum(axis=2).mean()
    comp2 = (-jnp.log(phi)).sum(axis=2).mean()
    return comp1, comp2


def weights_loss_all_starts(alpha, sigma_0, W_batch):
    """Average weights_loss_batched across all K cyclic start positions."""
    K = W_batch.shape[2]
    comp1_total = 0.0
    comp2_total = 0.0

    for start_idx in range(K):
        c1, c2 = weights_loss_batched(alpha, sigma_0, W_batch, start_idx=start_idx)
        comp1_total += c1
        comp2_total += c2

    return comp1_total / K, comp2_total / K


class Autoencoder(nnx.Module):
    def __init__(self, input_dim, n_components, rngs):
        self.encoder = nnx.Linear(input_dim, n_components, rngs=rngs, use_bias=False)
        self.decoder = nnx.Linear(n_components, input_dim, rngs=rngs, use_bias=False)
    
    def __call__(self, x):
        codes = self.encoder(x)
        latent = codes[..., None] * self.decoder.kernel[None, ...]
        recon = jnp.sum(latent, axis=1)
        latent_perm = jnp.transpose(latent, (0, 2, 1)) # [batch, columns, components]
        return recon, codes, latent_perm


def train_baseline(
    X,
    n_components,
    lr=1e-3,
    epochs=2000,
    batch_size=256,
    verbose=True,
    seed=42,
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
    
    n_samples, input_dim = X.shape
    X_jax = jnp.array(X, dtype=jnp.float32)
    
    # Initialize model
    rngs = nnx.Rngs(seed)
    model = Autoencoder(input_dim, n_components, rngs)
    
    # Initialize optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    
    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            recon, codes, _ = model(batch)
            return recon_loss(batch, recon, 1.0)  # sigma_eps=1 for baseline
        
        loss, grads = nnx.value_and_grad(loss_fn, has_aux=False)(model)
        optimizer.update(model, grads)
        return loss
    
    # Training loop
    key = random.PRNGKey(seed)
    
    for epoch in range(epochs):
        # Shuffle data
        key, subkey = random.split(key)
        idx = random.permutation(subkey, n_samples)
        permuted_X = X_jax[idx]
        
        epoch_losses = []
        
        for i in range(0, n_samples, batch_size):
            batch = permuted_X[i:i + batch_size]
            loss = train_step(model, optimizer, batch)
            epoch_losses.append(loss)
        
        if verbose and epoch % 200 == 0:
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            print(f"baseline epoch {epoch:4d} | recon_loss {float(avg_loss):.4f}")
    
    # Final evaluation
    recon, codes, _ = model(X_jax)
    final_loss = jnp.mean((X_jax - recon) ** 2)
    
    return SingleRun(
        model,
        np.array(codes),
        np.array(model.decoder.kernel),
        np.array(recon),
        float(final_loss),
        {},
    )




def train(
    X,
    n_components,
    lr=1e-2,
    epochs=2000,
    batch_size=64,
    verbose=True,
    init_strategy: InitStrategy=StandardInitStrategy(),
    initialised_model=None,
    use_ln_term=False,
    sigma_s_rel_to_0="equal",
    device="cpu",  # Ignored in JAX, kept for API compatibility
    baseline_epochs=1000,
    sigma_eps_override=None,
    recon_err_schedule=CosineAnnealReconError(1000),
    optim_type=AdamOptimType(),  # Only Adam supported
    sched_type=NoSchedType(),  # Schedulers not supported
    weights_algo: Literal["cyclic", "random"] = "random",
    seed=42,
) -> SingleRun:
    """
    JAX implementation of sparse autoencoder training.
    
    Args:
        X: numpy array (n_samples, input_dim)
        n_components: number of dictionary atoms
        device: Ignored in JAX (kept for API compatibility)
        sched_type: Must be NoSchedType() (schedulers not supported)
        optim_type: Must be AdamOptimType() (only Adam supported)
        seed: Random seed for JAX
        
    Note: Only StandardInitStrategy supported. WarmupInitStrategy will raise error.
    """
    # Parameter validation
    if not isinstance(init_strategy, StandardInitStrategy):
        raise ValueError("Only StandardInitStrategy supported in JAX implementation")
    
    if not isinstance(optim_type, AdamOptimType):
        raise ValueError("Only AdamOptimType supported in JAX implementation")
        
    if not isinstance(sched_type, NoSchedType):
        raise ValueError("Learning rate schedulers not supported in JAX implementation")
    
    if device != "cpu":
        print(f"Warning: device='{device}' ignored in JAX, using default device")
    
    print("using hyperparameters")
    print("\tinit_strategy", init_strategy)
    print("\tuse_ln", use_ln_term)
    print("\tsigma_s_rel_to_0", sigma_s_rel_to_0)
    
    X_jax = jnp.array(X, dtype=jnp.float32)
    n_samples, input_dim = X_jax.shape

    sigma_eps, baseline_loss = get_inferred_sigma_eps_if_needed(
        X, n_components, lr, baseline_epochs, batch_size, sigma_eps_override, verbose, train_baseline, seed=seed
    )

    if initialised_model is None:
        rngs = nnx.Rngs(seed)
        model = Autoencoder(input_dim, n_components, rngs)
        
        # Initialize using existing strategy (convert to JAX)
        scaled_hyperparameters = get_hyperparameters_and_init(
            model, X, n_components, input_dim, sigma_eps, init_strategy, sigma_s_rel_to_0, "cpu"
        )
    else:
        print("using initialised model")
        model = initialised_model
        scaled_hyperparameters = get_scaled_hyperparamters(
            X.std(),
            input_dim,
            n_components,
            sigma_eps=sigma_eps,
            w_to_eps_ratio=5,
            alpha_constant=5000,
            sigma_s_rel_to_0=sigma_s_rel_to_0,
        )

    best_model_state = None
    best_recon_loss = float("inf")

    p = scaled_hyperparameters
    sigma_x = X.std()
    sigma_eps, sigma_0, sigma_s, _, alpha = (
        p["sigma_eps"],
        p["sigma_0"],
        p["sigma_s"],
        p["sigma_enc"],
        p["alpha"],
    )
    print("using hyperparameters:", p)

    # Initialize optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch, recon_err_multiplier, epoch_mod):
        def loss_fn(model):
            recon, codes, latent_perm = model(batch)
            _recon_loss = recon_loss(batch, recon, sigma_eps)
            _recon_loss *= recon_err_multiplier
            
            if weights_algo == "random":
                comp1, comp2 = weights_loss_batched(alpha, sigma_x, latent_perm, epoch_mod)
            else:
                comp1, comp2 = weights_loss_all_starts(alpha, sigma_x, latent_perm)
            
            if use_ln_term:
                weight_loss = comp1 + comp2
            else:
                weight_loss = comp1
                
            codes_loss = gauss_loss(codes, 0) / (sigma_s * sigma_s)
            loss = _recon_loss + weight_loss + codes_loss
            
            return loss, (_recon_loss, weight_loss, codes_loss)
        
        (loss, (recon_loss_val, weight_loss_val, codes_loss_val)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        
        # Check for NaN
        if jnp.isnan(recon_loss_val) or jnp.isnan(weight_loss_val) or jnp.isnan(codes_loss_val):
            print(f"NaN detected: recon={recon_loss_val:.4f}, weight={weight_loss_val:.4f}, codes={codes_loss_val:.4f}")
            return loss, recon_loss_val, weight_loss_val, codes_loss_val, False  # Signal to break
            
        optimizer.update(model, grads)
        return loss, recon_loss_val, weight_loss_val, codes_loss_val, True

    # Training loop
    key = random.PRNGKey(seed)
    last_print_time = time.time()
    total_batches = math.ceil(n_samples / batch_size)

    for epoch in range(epochs):
        # Shuffle
        key, subkey = random.split(key)
        idx = random.permutation(subkey, n_samples)
        permuted_X_jax = X_jax[idx]

        epoch_recon_loss = []
        recon_err_multiplier = recon_err_schedule.get_multiplier(epoch, epochs)

        for i in range(0, n_samples, batch_size):
            batch = permuted_X_jax[i:i + batch_size]
            epoch_mod = epoch % n_components if weights_algo == "random" else 0
            
            loss, _recon_loss, weight_loss, codes_loss, should_continue = train_step(
                model, optimizer, batch, recon_err_multiplier, epoch_mod
            )
            
            if not should_continue:
                break
                
            epoch_recon_loss.append(float(_recon_loss))

        epoch_recon_loss_avg = np.mean(epoch_recon_loss)
        if epoch_recon_loss_avg < best_recon_loss:
            best_recon_loss = epoch_recon_loss_avg
            # Store best model state (functional copy)
            best_model_state = {k: jnp.array(v) for k, v in nnx.state(model).items()}

        if verbose and epoch % 50 == 0:
            print(
                f"epoch {epoch:4d} | recon_loss {float(_recon_loss):.4f} weight_loss {float(weight_loss):.4f} codes_loss {float(codes_loss):.4f} multiplier {recon_err_multiplier} duration={time.time() - last_print_time}"
            )
            last_print_time = time.time()

    if best_model_state is not None:
        print("loading the best model, recon loss", best_recon_loss)
        # Restore best state
        nnx.update(model, best_model_state)

    # Final evaluation
    recon, codes, _ = model(X_jax)

    return SingleRun(
        model,
        np.array(codes),
        np.array(model.decoder.kernel.T),  # Transpose to match PyTorch convention
        np.array(recon),
        float(jnp.mean((X_jax - recon) ** 2)),
        scaled_hyperparameters,
        baseline_loss,
    )