import torch
from torch import nn
from pt_to_api.benchmark.init_strats import (
    InitStrategy,
    NoInitStrategy,
    SvdInitStrategy,
    IcaInitStrategy,
    StandardInitStrategy,
    OnlyInitEncoderStrategy,
)
import numpy as np
from pt_to_api.benchmark.hyperparams import get_scaled_hyperparamters
from sklearn.decomposition import FastICA


def get_hyperparameters_and_init(
    model,
    X,
    n_components,
    input_dim,
    sigma_eps,
    init_strat: InitStrategy = NoInitStrategy(),
    sigma_s_rel_to_0="equal",
    device="cpu",
):
    sigma_x = X.std()
    scaled_hyperparameters = get_scaled_hyperparamters(
        sigma_x,
        input_dim,
        n_components,
        sigma_eps=sigma_eps,
        w_to_eps_ratio=5,
        alpha_constant=5000,
        sigma_s_rel_to_0=sigma_s_rel_to_0,
    )
    if isinstance(init_strat, SvdInitStrategy):
        init_model_parameters_using_svd(
            model, X, n_components, scaled_hyperparameters, device
        )
    elif isinstance(init_strat, IcaInitStrategy):
        init_model_parameters_using_ica(
            model, X, n_components, scaled_hyperparameters, device, init_strat.iters
        )
    elif isinstance(init_strat, StandardInitStrategy):
        init_model_parameters_using_normal(model, scaled_hyperparameters, n_components)
    elif isinstance(init_strat, OnlyInitEncoderStrategy):
        print("will only initialise the encoder with normal distribution")
        init_encoder_using_normal(model, scaled_hyperparameters["sigma_enc"])
    return scaled_hyperparameters


def init_model_parameters_using_svd(
    model, X, n_components, scaled_hyperparameters, device
):
    p = scaled_hyperparameters

    # svd
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    w = torch.tensor(Vt[:n_components], dtype=torch.float32).to(device)
    w = w.T
    # scale
    w = (w / w.std()) * p["sigma_0"]
    model.decoder.weight.data = w

    # encoder weights
    init_encoder_using_normal(model, p["sigma_enc"])


def init_encoder_using_normal(model, sigma_enc):
    # encoder weights
    nn.init.normal_(model.encoder.weight, mean=0.0, std=sigma_enc)
    # nn.init.zeros_(model.encoder.bias)


def init_model_parameters_using_normal(model, scaled_hyperparameters, n_components):
    p = scaled_hyperparameters
    sigma_0, sigma_enc = p["sigma_0"], p["sigma_enc"]

    # decoder = W, init with sigma_0
    nn.init.normal_(model.decoder.weight, mean=0.0, std=sigma_0)
    # nn.init.normal_(model.decoder.weight, mean=0.0, std=sigma_0)
    init_encoder_using_normal(model, sigma_enc)


def init_model_parameters_using_ica(
    model, X, n_components, scaled_hyperparameters, device, ica_iters
):
    p = scaled_hyperparameters

    ica_estimator = FastICA(
        n_components=n_components,
        max_iter=ica_iters,
        whiten="arbitrary-variance",
        tol=15e-5,
    )
    ica_estimator.fit(X)
    w = torch.tensor(ica_estimator.components_.T, dtype=torch.float32).to(device)
    w = (w / w.std()) * (p["sigma_0"] / np.sqrt(n_components))
    model.decoder.weight.data = w

    # encoder weights
    init_encoder_using_normal(model, p["sigma_enc"])
