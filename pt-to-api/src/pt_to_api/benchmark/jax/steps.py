import jax
from flax import nnx

from pt_to_api.benchmark.jax.core import ModelTrainStepUnconditionalParams
from pt_to_api.benchmark.jax.losses import (
    compute_baseline_loss,
    compute_full_loss,
    main_recon_loss_fn,
    main_weight_loss_fn,
)


@nnx.jit(static_argnames=["use_ln_term", "weights_algo"])
@nnx.vmap(in_axes=(0, None, 0, None, None, None, None), out_axes=0)
def parallel_eval_step(
    model, batch, uncond_params, use_ln_term, weights_algo, alpha_multiplier, epoch_mod
):
    loss, (loss_result, _) = compute_full_loss(
        model,
        batch,
        uncond_params,
        use_ln_term,
        weights_algo,
        alpha_multiplier,
        epoch_mod,
    )
    return loss, loss_result


@nnx.jit(static_argnames=["use_ln_term", "weights_algo"])
@nnx.vmap(in_axes=(0, 0, None, 0, None, None, None, None), out_axes=0)
def parallel_train_step(
    model,
    optimizer,
    batch,
    uncond_params: ModelTrainStepUnconditionalParams,
    use_ln_term,
    weights_algo,
    alpha_multiplier,
    epoch_mod,
):
    (_, (_, model_output)), grads = nnx.value_and_grad(compute_full_loss, has_aux=True)(
        model,
        batch,
        uncond_params,
        use_ln_term,
        weights_algo,
        alpha_multiplier,
        epoch_mod,
    )
    _, recon_grads = nnx.value_and_grad(main_recon_loss_fn, has_aux=True)(
        model, batch, uncond_params
    )
    _, weight_grads = nnx.value_and_grad(main_weight_loss_fn, has_aux=False)(
        model,
        batch,
        uncond_params,
        use_ln_term,
        weights_algo,
        alpha_multiplier,
        epoch_mod,
    )
    optimizer.update(model, grads)
    return grads, recon_grads, weight_grads


@nnx.jit
def baseline_train_step(model, optimizer, metrics, batch):
    loss, grads = nnx.value_and_grad(compute_baseline_loss)(model, batch)
    optimizer.update(model, grads)
    metrics.update(loss=loss)
    return loss
