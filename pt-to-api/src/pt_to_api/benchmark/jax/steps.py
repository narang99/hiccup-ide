from pt_to_api.benchmark.jax.core import ModelTrainStepUnconditionalParams
from pt_to_api.benchmark.jax.losses import compute_baseline_loss, compute_full_loss
from flax import nnx


@nnx.jit(static_argnames=["use_ln_term", "weights_algo"])
@nnx.vmap(in_axes=(0, 0, None, None, None, None), out_axes=0)
def parallel_eval_step(model, metrics, batch, uncond_params, use_ln_term, weights_algo):
    loss, loss_result = compute_full_loss(
        model,
        batch,
        uncond_params,
        use_ln_term,
        weights_algo,
    )
    metrics.update(
        loss=loss,
        recon_loss=loss_result.recon_loss,
        weight_loss=loss_result.weight_loss,
        codes_loss=loss_result.codes_loss,
        mse=loss_result.unscaled_mse,
    )
    return loss, loss_result


@nnx.jit(static_argnames=["use_ln_term", "weights_algo"])
@nnx.vmap(in_axes=(0, 0, None, None, None, None), out_axes=0)
def parallel_train_step(
    model,
    optimizer,
    batch,
    uncond_params: ModelTrainStepUnconditionalParams,
    use_ln_term,
    weights_algo,
):
    _, grads = nnx.value_and_grad(compute_full_loss, has_aux=True)(
        model, batch, uncond_params, use_ln_term, weights_algo
    )
    optimizer.update(model, grads)


@nnx.jit
def baseline_train_step(model, optimizer, metrics, batch):
    loss, grads = nnx.value_and_grad(compute_baseline_loss)(model, batch)
    optimizer.update(model, grads)
    metrics.update(loss=loss)
    return loss
