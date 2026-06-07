from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from pt_to_api.benchmark.jax.core import ModelTrainStepUnconditionalParams
from pt_to_api.benchmark.losses import get_gauss_loss, get_recon_loss


def weights_loss_batched(alpha, W_batch, start_idx=0):
    """Batched version without cyclic shifts. W_batch: [B, C, K]"""
    W_rolled = jnp.roll(W_batch, -start_idx, axis=2)
    W_sq = W_rolled**2
    cumsum = jnp.cumsum(W_sq, axis=2)
    phi = alpha * jnp.roll(cumsum, 1, axis=2) + 1
    phi = phi.at[:, :, 0].set(1)
    # comp1 = (W_sq * phi).sum(axis=2).mean()
    # comp2 = (-jnp.log(phi)).sum(axis=2).mean()
    comp1 = (W_sq * phi).mean()
    comp2 = (-jnp.log(phi)).mean()
    return comp1, comp2


def weights_loss_correlation_matrix(latent):
    # [batch, columns, components] is latents shape
    # we want to find the correlation matrix
    # [components, columns] -> [components, columns] -> [components, components]
    # mult each componet dot prod to every other comp. tis easy
    # jnp.einsum("bkd,bk->b")
    corr_mat = jnp.einsum("bdi,bdj->bij", latent, latent)
    tot_loss = jnp.triu(corr_mat, k=1)
    return tot_loss.mean()


def weights_loss_all_starts(alpha, W_batch):
    """Average weights_loss_batched across all K cyclic start positions."""
    K = W_batch.shape[2]
    comp1_total = 0.0
    comp2_total = 0.0

    for start_idx in range(K):
        c1, c2 = weights_loss_batched(alpha, W_batch, start_idx=start_idx)
        comp1_total += c1
        comp2_total += c2

    return comp1_total / K, comp2_total / K


class EvalResult(NamedTuple):
    recon: jnp.ndarray
    codes: jnp.ndarray


class LossResult(NamedTuple):
    eval_result: EvalResult
    recon_loss: jnp.ndarray
    codes_loss: jnp.ndarray
    weight_loss: jnp.ndarray
    unscaled_mse: jnp.ndarray


def _get_weight_loss(
    latent_perm,
    uncond_params: ModelTrainStepUnconditionalParams,
    use_ln_term,
    weights_algo,
    alpha_multiplier,
    epoch_mod,
):
    u = uncond_params
    alpha = u.alpha * alpha_multiplier
    if weights_algo == "random":
        comp1, comp2 = weights_loss_batched(alpha, latent_perm, epoch_mod)
        if use_ln_term:
            weight_loss = comp1 + comp2
        else:
            weight_loss = comp1
    elif weights_algo == "corr":
        weight_loss = weights_loss_correlation_matrix(latent_perm)
    else:
        comp1, comp2 = weights_loss_all_starts(alpha, latent_perm)
        if use_ln_term:
            weight_loss = comp1 + comp2
        else:
            weight_loss = comp1

    return weight_loss * u.weights_coeff


def main_weight_loss_fn(
    model, batch, uncond_params, use_ln_term, weights_algo, alpha_multiplier, epoch_mod
):
    recon, codes, latent = model(batch)
    return _get_weight_loss(
        latent, uncond_params, use_ln_term, weights_algo, alpha_multiplier, epoch_mod
    )


def _get_recon_loss(batch, recon, uncond_params: ModelTrainStepUnconditionalParams):
    u = uncond_params
    unscaled_mse = get_gauss_loss(batch, recon)
    _recon_loss = unscaled_mse * u.recon_coeff
    return _recon_loss, unscaled_mse


def main_recon_loss_fn(model, batch, uncond_params):
    recon, _, _ = model(batch)
    return _get_recon_loss(batch, recon, uncond_params)


def compute_full_loss(
    model: nnx.Module,
    batch,
    uncond_params: ModelTrainStepUnconditionalParams,
    use_ln_term,
    weights_algo,
    alpha_multiplier,
    epoch_mod,
) -> tuple[jnp.ndarray, tuple[LossResult, dict]]:
    """Pure function to compute full loss with all components."""
    u = uncond_params

    recon, codes, latent_perm = model(batch)
    # alpha, recon_err_multiplier, epoch_mod = uncond_params.alpha, uncond_params.recon_err_multiplier, uncond_params.epoch_mod
    # sigma_eps, sigma_0, sigma_s = uncond_params.sigma_eps, uncond_params.sigma_0, uncond_params.sigma_s

    _recon_loss, unscaled_mse = _get_recon_loss(batch, recon, uncond_params)

    weight_loss = _get_weight_loss(
        latent_perm,
        uncond_params,
        use_ln_term,
        weights_algo,
        alpha_multiplier,
        epoch_mod,
    )

    codes_loss = u.recon_coeff * get_gauss_loss(codes, 0)
    loss = _recon_loss + weight_loss + codes_loss

    loss_result = LossResult(
        eval_result=EvalResult(recon, codes),
        recon_loss=_recon_loss,
        codes_loss=codes_loss,
        weight_loss=weight_loss,
        unscaled_mse=unscaled_mse,
    )

    model_output = {
        "recon": recon,
        "latent": latent_perm,
    }

    return loss, (loss_result, model_output)


def compute_baseline_loss(model, batch):
    """Pure function to compute baseline reconstruction loss."""
    recon, _, _ = model(batch)
    return get_recon_loss(batch, recon, 1.0)  # sigma_eps=1 for baseline
