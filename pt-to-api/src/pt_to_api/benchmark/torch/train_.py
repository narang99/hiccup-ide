"""
The constraint is slightly different now.

Instead of putting the constraint on W, we put it on S*W. Easy. it actually works quite well in practice surprisingly.
"""

from functools import partial
import time
from torch import nn
import torch
from torch import optim
from pt_to_api.benchmark.core import SingleRun
from pt_to_api.benchmark.init_strats import (
    InitStrategy,
    StandardInitStrategy,
)
from pt_to_api.benchmark.anneal import (
    CosineAnnealReconError,
    ReconErrSchedule,
)
from pt_to_api.benchmark.torch.model_init import get_hyperparameters_and_init
from pt_to_api.benchmark.losses import (
    get_recon_loss,
    get_gauss_loss,
)
from pt_to_api.benchmark.hyperparams import (
    get_scaled_hyperparamters,
    get_inferred_sigma_eps_if_needed,
)
import numpy as np
from dataclasses import dataclass
from typing import Literal
import math


class Autoencoder(nn.Module):
    def __init__(self, input_dim, n_components):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_components, bias=False)
        self.decoder = nn.Linear(n_components, input_dim, bias=False)

    def forward(self, x):
        codes = self.encoder(x)
        latent = codes.unsqueeze(-1) * self.decoder.weight.T.unsqueeze(0)
        recon = latent.sum(dim=1)
        latent_perm = latent.permute(0, 2, 1)  # [batch, dims, n_components]
        return recon, codes, latent_perm


@dataclass
class CosineAnnealingWithWarmRestartsSchedType:
    T_0: int
    T_mult: int
    eta_min: float


@dataclass
class NoSchedType:
    pass


@dataclass
class AdamOptimType:
    pass


@dataclass
class SGDOptimType:
    pass


OptimType = SGDOptimType | AdamOptimType
SchedType = NoSchedType | CosineAnnealingWithWarmRestartsSchedType


def train(
    X,
    n_components,
    lr=1e-2,
    epochs=2000,
    batch_size=64,
    verbose=True,
    init_strategy: InitStrategy = StandardInitStrategy(),
    initialised_model=None,
    use_ln_term=False,
    sigma_s_rel_to_0="equal",
    device="cpu",
    baseline_epochs=1000,
    sigma_eps_override=None,
    recon_err_schedule: ReconErrSchedule = CosineAnnealReconError(1000),
    optim_type: OptimType = AdamOptimType(),
    sched_type: SchedType = NoSchedType(),
    weights_algo: Literal["cyclic", "random"] = "random",
) -> SingleRun:
    """
    X: numpy array (n_samples, input_dim)
    n_components: number of dictionary atoms
    alpha: lorentzian sigma shrinker parameter
    """
    print("using hyperparameters")
    print("\tinit_strategy", init_strategy)
    print("\tuse_ln", use_ln_term)
    print("\tsigma_s_rel_to_0", sigma_s_rel_to_0)

    X_t = torch.tensor(X, dtype=torch.float32)
    n_samples, input_dim = X_t.shape
    X_t = X_t.to(device)

    sigma_eps, baseline_loss = get_inferred_sigma_eps_if_needed(
        X,
        n_components,
        lr,
        baseline_epochs,
        batch_size,
        sigma_eps_override,
        verbose,
        partial(train_baseline, device=device),
    )
    if initialised_model is None:
        model = Autoencoder(input_dim, n_components)
    else:
        print("using initialised model")
        model = initialised_model

    model = model.to(device)
    if initialised_model is None:
        # only initialise if needed
        scaled_hyperparameters = get_hyperparameters_and_init(
            model,
            X,
            n_components,
            input_dim,
            sigma_eps,
            init_strategy,
            sigma_s_rel_to_0,
            device,
        )
    else:
        print("skipping model initialised, found already initialised model")
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

    if isinstance(optim_type, AdamOptimType):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif isinstance(optim_type, SGDOptimType):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise Exception(f"unknown optimiser type: {optim_type}")

    scheduler = None
    if isinstance(sched_type, NoSchedType):
        # do not set the scheduler
        pass
    elif isinstance(sched_type, CosineAnnealingWithWarmRestartsSchedType):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, sched_type.T_0, sched_type.T_mult, sched_type.eta_min
        )
    else:
        raise Exception(f"unknown sched type: {sched_type}")

    last_print_time = time.time()
    total_batches = math.ceil(n_samples / batch_size)

    for epoch in range(epochs):
        # shuffle
        idx = torch.randperm(n_samples, device=device)
        permuted_X_t = X_t[idx]

        epoch_recon_loss = []
        recon_err_multiplier = recon_err_schedule.get_multiplier(epoch, epochs)

        for i in range(0, n_samples, batch_size):
            batch = permuted_X_t[i : i + batch_size]
            recon, codes, latent_perm = model(batch)
            _recon_loss = get_recon_loss(batch, recon, sigma_eps)
            _recon_loss *= recon_err_multiplier
            if weights_algo == "random":
                comp1, comp2 = weights_loss_batched(
                    alpha, sigma_x, latent_perm, epoch % latent_perm.shape[2]
                )
            else:
                comp1, comp2 = weights_loss_all_starts(alpha, sigma_x, latent_perm)
            if use_ln_term:
                weight_loss = comp1 + comp2
            else:
                weight_loss = comp1
            codes_loss = get_gauss_loss(codes, 0) / (sigma_s * sigma_s)
            loss = _recon_loss + weight_loss + codes_loss
            if (
                torch.isnan(_recon_loss)
                or torch.isnan(weight_loss)
                or torch.isnan(codes_loss)
            ):
                print(
                    f"NaN detected: recon={_recon_loss:.4f}, weight={weight_loss:.4f}, codes={codes_loss:.4f}"
                )
                break
            optimizer.zero_grad()
            loss.backward()
            if isinstance(optimizer, optim.SGD):
                # SGD struggles with huge gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + i / total_batches)
            epoch_recon_loss.append(_recon_loss.item())

        epoch_recon_loss = np.mean(epoch_recon_loss)
        if epoch_recon_loss < best_recon_loss:
            best_recon_loss = epoch_recon_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 50 == 0:
            print(
                f"epoch {epoch:4d} | recon_loss {_recon_loss:.4f} weight_loss {weight_loss:.4f} codes_loss {codes_loss:.4f} multiplier {recon_err_multiplier} duration={time.time() - last_print_time}"
            )
            last_print_time = time.time()

    if best_model_state is not None:
        print("loading the best model, recon loss", best_recon_loss)
        model.load_state_dict(best_model_state)

    with torch.no_grad():
        recon, codes, _ = model(X_t)

    # decoder shape: [dimensions, n-components]
    # encoder shape: [n-components, dimensions]
    # single-run wants [n-components, dimensions] for both
    return SingleRun(
        codes=codes.to("cpu").numpy(),
        encoder=model.encoder.weight.detach().to("cpu").numpy(),
        components=model.decoder.weight.T.detach().to("cpu").numpy(),
        recon=recon.to("cpu").numpy(),
        loss=((X_t - recon) ** 2).to("cpu").mean(),
        scaled_hyperparameters=scaled_hyperparameters,
        baseline_loss=baseline_loss,
    )


def train_baseline(
    X,
    n_components,
    lr=1e-3,
    epochs=2000,
    batch_size=256,
    verbose=True,
    device="cpu",
):
    """Train the autoencoder with just reconstruction loss, to find an arbitrary linear model which fits the data

    The main training code requires sigma_eps
    the standard deviation of expected gaussian noise
    when the curve is fitted using Y=WX
    We can generally do a simple sweep of hyperparams
    or use simple heuristics
    If the data is linearly "fittable",
    then we get a good starting point
    using this function.
    """
    print(f"training baseline model, epochs={epochs} device={device}")
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    n_samples, input_dim = X_t.shape

    model = Autoencoder(input_dim, n_components).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        idx = torch.randperm(n_samples, device=device)
        permuted_X_t = X_t[idx]
        for i in range(0, n_samples, batch_size):
            batch = permuted_X_t[i : i + batch_size]
            recon, codes, _ = model(batch)
            _recon_loss = get_recon_loss(batch, recon, 1)
            loss = _recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and epoch % 200 == 0:
            print(f"finetune epoch {epoch:4d} | recon_loss {_recon_loss:.4f}")

    with torch.no_grad():
        recon, codes, _ = model(X_t)

    # decoder shape: [dimensions, n-components]
    # encoder shape: [n-components, dimensions]
    # single-run wants [n-components, dimensions] for both
    return SingleRun(
        codes=codes.to("cpu").numpy(),
        encoder=model.encoder.weight.detach().to("cpu").numpy(),
        components=model.decoder.weight.T.detach().to("cpu").numpy(),
        recon=recon.to("cpu").numpy(),
        loss=((X_t - recon) ** 2).mean().item(),
        hyperparameters={},
    )


def weights_loss_batched(alpha, sigma_0, W_batch, start_idx=0):
    """Batched version without cyclic shifts. W_batch: [B, C, K]"""
    W_rolled = torch.roll(W_batch, -start_idx, dims=2)
    W_sq = W_rolled**2  # (B, C, K)
    cumsum = torch.cumsum(W_sq, dim=2)  # (B, C, K)
    phi = alpha * torch.roll(cumsum, 1, dims=2) + 1  # (B, C, K)
    phi[:, :, 0] = 1
    comp1 = (W_sq * phi / (sigma_0 * sigma_0)).sum(dim=2).mean()
    comp2 = (-torch.log(phi)).sum(dim=2).mean()
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


def error_is_gauss_with_sigma_loss(batch, x, sigma_eps):
    residual = x - batch
    # batch: n_samples, dims
    # we need to create a random vector of 20 dim for each sample
    # subtract normal dist from this, sampled from sigma_eps
    gauss_residuals = torch.randn(
        (20, *batch.shape), dtype=torch.float32, device=batch.device
    )
    residual_for_each_sample = (residual - gauss_residuals).sum(dim=0).sum(dim=-1)
    # print("residual for each sample", residual_for_each_sample)
    return residual_for_each_sample.abs().sum()
