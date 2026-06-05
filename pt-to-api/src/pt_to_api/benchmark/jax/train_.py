import gc
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tensorboardX.writer import SummaryWriter

from pt_to_api.benchmark.anneal import (
    CosineAnnealReconError,
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
    sigma_s_rel_to_0="equal",
    baseline_epochs=1000,
    sigma_eps_override=None,
    recon_err_schedule: ReconErrSchedule = CosineAnnealReconError(1000),
    weights_algo: Literal["cyclic", "random"] = "random",
    seed=42,
    eval_every=200,
    tensorboard_log_dir=None,
) -> list[SingleRun]:
    print(f"Training {n_models} models")

    # Parameter validation (same as single version)
    if not isinstance(init_strategy, (StandardInitStrategy, NoInitStrategy)):
        raise ValueError("Only StandardInitStrategy and NoInitStrategy supported")

    x_jax = jnp.array(X, dtype=jnp.float32)
    n_samples, input_dim = x_jax.shape
    print("total training samples", n_samples)

    p = get_scaled_hyperparameters_after_inferring_sigma_eps(
        X,
        n_components,
        lr,
        baseline_epochs,
        batch_size,
        sigma_eps_override,
        sigma_s_rel_to_0,
        verbose,
        # baseline train is deterministic, we dont care, just need sigma_eps
        nnx.Rngs(0),
    )
    print("shared scaled_hyperparameters", p)

    # first key used for the main training code
    # remaining used for model initialisation
    all_keys = jax.random.split(jax.random.PRNGKey(seed), n_models + 1)
    key, keys = all_keys[0], all_keys[1:]

    models, optimizers = parallel_init_models(
        keys, input_dim, n_components, p, init_strategy, lr
    )
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

    get_uncond = partial(
        ModelTrainStepUnconditionalParams,
        sigma_eps=p["sigma_eps"],
        sigma_0=p["sigma_0"],
        sigma_s=p["sigma_s"],
        alpha=p["alpha"],
    )

    last_print_time = time.time()
    for epoch in range(epochs):
        recon_err_multiplier = recon_err_schedule.get_multiplier(epoch, epochs)
        epoch_mod = epoch % n_components if weights_algo == "random" else 0
        uncond_params = get_uncond(
            recon_err_multiplier=recon_err_multiplier, epoch_mod=epoch_mod
        )

        for key, batch in get_batches(x_jax, batch_size, key):
            parallel_train_step(
                models, optimizers, batch, uncond_params, use_ln_term, weights_algo
            )

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
            )
            log_to_tensorboard(writers, metrics, n_components, epoch)
            best_model_manager.update_and_ckpt(models, metrics, "mse")
            print(f"epoch {epoch} | duration = {time.time() - last_print_time}")
            last_print_time = time.time()

    return get_single_runs(best_model_manager, x_jax, n_components, p)


def do_eval_steps(
    x_jax, models, metrics, uncond_params, use_ln_term, weights_algo, batch_size: int
):
    for metric in metrics:
        metric.reset()
    for _, batch in get_batches(x_jax, batch_size, None):
        loss, loss_result = parallel_eval_step(
            models, batch, uncond_params, use_ln_term, weights_algo
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
    hyperparameters: dict,
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
            hyperparameters=hyperparameters,
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
