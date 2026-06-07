import gc
import math
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict
from tensorboardX.writer import SummaryWriter

from pt_to_api.benchmark.anneal import (
    ConstantReconError,
    ReconErrSchedule,
)
from pt_to_api.benchmark.core import SingleRun
from pt_to_api.benchmark.hyperparams import (
    get_inferred_sigma_eps_if_needed,
    get_scaled_hyperparamters,
)
from pt_to_api.benchmark.init_strats import (
    InitStrategy,
    NoInitStrategy,
    StandardInitStrategy,
)
from pt_to_api.benchmark.jax.ckpt import BestModelManager
from pt_to_api.benchmark.jax.core import (
    Autoencoder,
    ModelTrainStepUnconditionalParams,
    get_batches,
)
from pt_to_api.benchmark.jax.init_models import parallel_init_models
from pt_to_api.benchmark.jax.steps import parallel_eval_step, parallel_train_step
from pt_to_api.benchmark.jax.train_baseline_ import train_baseline


def train(
    X,
    n_components,
    n_models=10,
    lr=1e-2,
    epochs=2000,
    batch_size=64,
    verbose=True,
    init_strategy: InitStrategy = StandardInitStrategy(),
    use_ln_term=False,
    alpha_schedule: ReconErrSchedule = ConstantReconError(),
    weights_algo: Literal["cyclic", "random"] = "random",
    seed=42,
    eval_every=200,
    tensorboard_log_dir=None,
    alpha_range=(5000.0, 10_000.0),
    recon_coeff_range=(10.0, 100.0),
    weights_coeff_range=(1.0, 10.0),
    codes_coeff_range=(1.0, 2.0),
) -> list[SingleRun]:
    print(f"Training {n_models} models")

    # Parameter validation (same as single version)
    if not isinstance(init_strategy, (StandardInitStrategy, NoInitStrategy)):
        raise ValueError("Only StandardInitStrategy and NoInitStrategy supported")

    x_jax = jnp.array(X, dtype=jnp.float32)
    n_samples, input_dim = x_jax.shape
    print("total training samples", n_samples)

    # first key used for the main training code
    # remaining used for model initialisation
    all_keys = jax.random.split(jax.random.PRNGKey(seed), n_models + 1)
    key, keys = all_keys[0], all_keys[1:]

    models, optimizers = parallel_init_models(keys, input_dim, n_components, lr)
    metrics = [
        nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            recon_loss=nnx.metrics.Average("recon_loss"),
            weight_loss=nnx.metrics.Average("weight_loss"),
            codes_loss=nnx.metrics.Average("codes_loss"),
            mse=nnx.metrics.Average("mse"),
        )
        for _ in range(n_models)
    ]
    key, subkey = jax.random.split(key)
    uncond_params = sample_hyperparameters(
        subkey,
        n_models,
        alpha_range,
        recon_coeff_range,
        codes_coeff_range,
        weights_coeff_range,
    )
    print("using uncond params", uncond_params)
    log_dir = (
        Path(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else Path("tensorboard/experiment")
    )
    print(f"Tensorboard log dir: {log_dir}")
    # we get separate log directory for every n_components
    # we remove them for this new run
    # all seeds use the same metric name (comps_{comp_name}.{metric_key})
    # thus they are overlaid on the same graph
    log_dirs = [log_dir / f"seed_{i}" for i in range(n_models)]
    for d in log_dirs:
        shutil.rmtree(d, ignore_errors=True)
    writers = [
        SummaryWriter(log_dir=str(log_dir / f"seed_{i}")) for i in range(n_models)
    ]
    best_model_manager = BestModelManager(n_models)
    last_print_time = time.time()

    for epoch in range(epochs):
        # recon_err_multiplier = recon_err_schedule.get_multiplier(epoch, epochs)
        epoch_mod = epoch % n_components if weights_algo == "random" else 0
        alpha_multiplier = alpha_schedule.get_multiplier(epoch, epochs)

        for key, batch in get_batches(x_jax, batch_size, key):
            # now we need to scalar it
            parallel_train_step(
                models,
                optimizers,
                batch,
                uncond_params,
                use_ln_term,
                weights_algo,
                alpha_multiplier,
                epoch_mod,
            )
            # log_grads_to_tensorboard(writers, grads, n_components, epoch)

        if verbose and epoch % eval_every == 0:
            # no shuffling in eval steps
            do_eval_steps(
                x_jax,
                models,
                metrics,
                uncond_params,
                use_ln_term,
                weights_algo,
                batch_size,
                alpha_multiplier,
                epoch_mod,
            )
            log_to_tensorboard(writers, metrics, n_components, epoch)
            best_model_manager.update_and_ckpt(models, metrics, "mse")
            print(f"epoch {epoch} | duration = {time.time() - last_print_time}")
            last_print_time = time.time()

    return get_single_runs(best_model_manager, x_jax, n_components)


def do_eval_steps(
    x_jax,
    models,
    metrics,
    uncond_params,
    use_ln_term,
    weights_algo,
    batch_size: int,
    alpha_multiplier,
    epoch_mod,
):
    for metric in metrics:
        metric.reset()
    for _, batch in get_batches(x_jax, batch_size, None):
        loss, loss_result = parallel_eval_step(
            models,
            batch,
            uncond_params,
            use_ln_term,
            weights_algo,
            alpha_multiplier,
            epoch_mod,
        )
        # update metrics
        for i in range(loss.shape[0]):
            metrics[i].update(
                loss=loss[i],
                recon_loss=loss_result.recon_loss[i],
                weight_loss=loss_result.weight_loss[i],
                codes_loss=loss_result.codes_loss[i],
                mse=loss_result.unscaled_mse[i],
            )


def log_grads_to_tensorboard(writers, grads, n_components, epoch):
    main_grads, recon_grads, weight_grads = grads
    log_one_grad_to_tensorboard(writers, main_grads, "main", n_components, epoch)
    log_one_grad_to_tensorboard(writers, weight_grads, "weight", n_components, epoch)
    log_one_grad_to_tensorboard(writers, recon_grads, "recon", n_components, epoch)


def log_one_grad_to_tensorboard(writers, grads, grad_id, n_components, epoch):
    flat_grads = flatten_dict(nnx.to_pure_dict(grads), sep="/")
    decoder_grads = flat_grads["decoder/kernel"].mean(axis=1).mean(axis=1)
    encoder_grads = flat_grads["encoder/kernel"].mean(axis=1).mean(axis=1)
    for writer, dec_grad, enc_grad in zip(writers, decoder_grads, encoder_grads):
        writer.add_scalar(
            f"grads_{n_components}.{grad_id}.decoder", float(dec_grad), epoch
        )
        writer.add_scalar(
            f"grads_{n_components}.{grad_id}.encoder", float(enc_grad), epoch
        )


def log_to_tensorboard(writers, metrics, n_components, epoch):
    for writer, metric in zip(writers, metrics):
        computed = metric.compute()
        for metric_key in ["loss", "recon_loss", "weight_loss", "codes_loss", "mse"]:
            writer.add_scalar(
                f"comps_{n_components}.{metric_key}",
                float(computed[metric_key]),
                epoch,
            )
            writer.flush()


def get_single_runs(
    best_model_manager: BestModelManager,
    X_jax: jnp.ndarray,
    n_components: int,
) -> list[SingleRun]:
    input_dim = X_jax.shape[1]
    decoders, encoders, models = (
        restore_best_encoder_and_decoder_weights_in_single_run_format(
            best_model_manager, input_dim, n_components
        )
    )
    all_recons, all_codes = [], []

    # run eval again
    for model in models:
        recon, codes, _ = model(X_jax)
        all_recons.append(np.array(recon))
        all_codes.append(np.array(codes))
        gc.collect()

    return [
        SingleRun(
            codes=all_codes[i],
            encoder=encoders[i],
            components=decoders[i],
            recon=all_recons[i],
            loss=float(best_model_manager.best_losses[i]),
            hyperparameters={},
        )
        for i in range(len(models))
    ]


def restore_best_encoder_and_decoder_weights_in_single_run_format(
    best_model_manager: BestModelManager, input_dim, n_components
):
    # best_model_states = best_model_manager.load_best_model_states()
    best_models = best_model_manager.load_best_models(
        nnx.eval_shape(lambda: Autoencoder(input_dim, n_components, nnx.Rngs(0)))
    )
    # Flax follows [input, output]
    # encoder: [dimensions, n_components]
    # decoder: [n_components, dimensions]
    # we want [n_components, dimensions] for both in run format
    decoders, encoders = [], []
    for m in best_models:
        decoders.append(np.array(m.decoder.kernel))
        encoders.append(np.array(m.encoder.kernel.T))
    return decoders, encoders, best_models


def get_scaled_hyperparameters_after_inferring_sigma_eps(
    X,
    n_components,
    lr,
    baseline_epochs,
    batch_size,
    sigma_eps_override,
    sigma_s_rel_to_0,
    verbose,
    rngs,
):
    input_dim = X.shape[1]
    # Get sigma_eps (same for all models)
    sigma_eps, _ = get_inferred_sigma_eps_if_needed(
        X,
        n_components,
        lr,
        baseline_epochs,
        batch_size,
        sigma_eps_override,
        verbose,
        partial(train_baseline, rngs=rngs),
    )

    # Get scaled hyperparameters once (shared by all models)
    scaled_hyperparameters = get_scaled_hyperparamters(
        X.std(),
        input_dim,
        n_components,
        sigma_eps=sigma_eps,
        w_to_eps_ratio=5,
        alpha_constant=5000,
        sigma_s_rel_to_0=sigma_s_rel_to_0,
    )
    return scaled_hyperparameters


def print_metrics_at_eval(epoch, metrics, last_print_time):
    avg_losses = [m.compute() for m in metrics]
    print(f"epoch {epoch:4d} | duration={time.time() - last_print_time:.2f}s")
    for i in range(len(avg_losses)):
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses[i].items())
        print(f"\tseed {i}: {parts}")


def sample_hyperparameters(
    key,
    batch_size,
    alpha_range,
    recon_coeff_range,
    codes_coeff_range,
    weights_coeff_range,
):
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    alphas = jax.random.uniform(
        rng1, shape=(batch_size,), minval=alpha_range[0], maxval=alpha_range[1]
    )
    recon_coeffs = jax.random.uniform(
        rng2,
        shape=(batch_size,),
        minval=recon_coeff_range[0],
        maxval=recon_coeff_range[1],
    )
    codes_coeffs = jax.random.uniform(
        rng3,
        shape=(batch_size,),
        minval=codes_coeff_range[0],
        maxval=codes_coeff_range[1],
    )
    weights_coeffs = jax.random.uniform(
        rng4,
        shape=(batch_size,),
        minval=weights_coeff_range[0],
        maxval=weights_coeff_range[1],
    )
    ln_terms = jnp.ones((batch_size,))
    return ModelTrainStepUnconditionalParams(
        alphas, recon_coeffs, codes_coeffs, weights_coeffs, ln_terms
    )
