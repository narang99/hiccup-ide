import time
from dataclasses import dataclass
from sklearn.decomposition import FastICA
import numpy as np
import torch
from torch import nn
from torch import optim
from .core import SingleRun
from .utils import (
    InitStrategy, 
    SvdInitStrategy, 
    WarmupInitStrategy, 
    IcaInitStrategy, 
    NoInitStrategy, 
    StandardInitStrategy, 
    OnlyInitEncoderStrategy,
)



class Autoencoder(nn.Module):
    def __init__(self, input_dim, n_components):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_components, bias=True)
        self.decoder = nn.Linear(n_components, input_dim, bias=False)

    def forward(self, x):
        codes = self.encoder(x)
        recon = self.decoder(codes)
        return recon, codes


def recon_loss(x, recons, sigma_eps):
    """Reconstruction loss. MSE"""
    return gauss_loss(x, recons) / (sigma_eps * sigma_eps)


def codes_loss(codes, sigma_s):
    """L2 loss on encoder"""
    return gauss_loss(codes, 0) / (sigma_s * sigma_s)


def gauss_loss(x, mean):
    loss = (x - mean) ** 2
    # return torch.sum(loss, 1).mean()
    return loss.mean()


def get_inferred_sigma_eps_if_needed(
    X, n_components, lr, baseline_epochs, batch_size, sigma_eps_override, verbose, train_baseline_fn, **train_baseline_kwargs
):
    """Framework-agnostic sigma_eps inference that works with any train_baseline function."""
    if sigma_eps_override is not None:
        print("sigma_eps_override passed, skipping baseline run")
        sigma_eps = sigma_eps_override
        baseline_loss = None
    else:
        baseline_run = train_baseline_fn(X, n_components, lr, baseline_epochs, batch_size, verbose, **train_baseline_kwargs)
        sigma_eps = np.sqrt(baseline_run.loss)
        tol = X.std() / 10_000
        if sigma_eps < tol:
            print(f"WARM: sigma_eps={sigma_eps} is less than tolerance={tol}, this can have undesired behavior")
            sigma_eps = tol
        print("baseline MSE", baseline_run.loss)
        baseline_loss = baseline_run.loss
    return sigma_eps, baseline_loss

def weights_loss_cycled(alpha, sigma_0, W):
    """Vectorized weight loss averaged over all starting positions."""
    C, K = W.shape
    W_sq = W ** 2  # (C, K)

    # Build all K cyclic shifts: (C, K, K)
    # shifts[c, s, :] = W_sq[c, :] rolled by -s
    idx = (torch.arange(K).unsqueeze(0) - torch.arange(K).unsqueeze(1)) % K  # (K, K)
    W_sq_shifted = W_sq[:, idx]  # (C, K, K)

    cumsum = torch.cumsum(W_sq_shifted, dim=2)             # (C, K, K)
    phi = alpha * torch.roll(cumsum, 1, dims=2) + 1        # (C, K, K)
    phi[:, :, 0] = 1

    comp1 = (W_sq_shifted * phi / (sigma_0 * sigma_0)).sum(dim=2)  # (C, K)
    comp2 = (-torch.log(phi)).sum(dim=2)                            # (C, K)

    # average over shifts (dim=1), then over channels (dim=0)
    return comp1.mean(), comp2.mean()


def weights_loss(alpha, sigma_0, W):
    """Vectorized version the Weight loss"""
    W_sq = W**2  # (C, K)
    cumsum = torch.cumsum(W_sq, dim=1)  # (C, K), cumsum[c,k] = sum W[c,0..k]^2
    phi = alpha * torch.roll(cumsum, 1, dims=1) + 1  # (C, K)
    phi[:, 0] = 1  # k=0: phi_weight(W, c, -1, alpha) = alpha*0 + 1
    comp1 = (W_sq * phi) / (sigma_0 * sigma_0)
    comp2 = -torch.log(phi)
    return comp1.sum(dim=1).mean(), comp2.sum(dim=1).mean()



def get_scaled_hyperparamters(
    sigma_x,
    input_dim,
    n_components,
    sigma_eps,
    w_to_eps_ratio=5,
    alpha_constant=10_000.0,
    sigma_s_rel_to_0="equal",
):
    """
    sigma_x      : std of your data
    eps_ratio    : sigma_x / sigma_eps (default 100)
    w_to_eps_ratio: sigma_0 / sigma_eps (default 5)
    alpha_constant: the c in alpha = c / sigma_0^2
    """
    if sigma_s_rel_to_0 == "equal":
        sigma_0 = sigma_s = sigma_x / np.sqrt(n_components)
        # sigma_0 = sigma_s = sigma_x
    elif sigma_s_rel_to_0 == "less":
        sigma_s = sigma_eps * w_to_eps_ratio
        sigma_0 = sigma_x / sigma_s 
    else:
        sigma_0 = sigma_eps * w_to_eps_ratio
        sigma_s = sigma_x / sigma_0

    sigma_enc = sigma_s / (
        np.sqrt(input_dim) * sigma_x
    )  # from D*sigma_enc^2*sigma_x^2 = sigma_s^2
    alpha = alpha_constant / (sigma_0**2)

    return dict(
        sigma_eps=sigma_eps,
        sigma_0=sigma_0,
        sigma_s=sigma_s,
        sigma_enc=sigma_enc,
        alpha=alpha,
    )


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


def init_model_parameters_using_svd(model, X, n_components, scaled_hyperparameters, device):
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


def init_model_parameters_using_ica(model, X, n_components, scaled_hyperparameters, device, ica_iters):
    p = scaled_hyperparameters

    ica_estimator = FastICA(
        n_components=n_components, max_iter=ica_iters, whiten="arbitrary-variance", tol=15e-5
    )
    ica_estimator.fit(X)
    w = torch.tensor(ica_estimator.components_.T, dtype=torch.float32).to(device)
    w = (w / w.std()) * (p["sigma_0"]/np.sqrt(n_components))
    print("ica w shapeeeee", w.shape, "model decoder shape", model.decoder.weight.shape)
    model.decoder.weight.data = w

    # encoder weights
    init_encoder_using_normal(model, p["sigma_enc"])


def get_hyperparameters_and_init(
    model, X, n_components, input_dim, sigma_eps, init_strat:InitStrategy=NoInitStrategy(), sigma_s_rel_to_0="equal", device="cpu"
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
        init_model_parameters_using_svd(model, X, n_components, scaled_hyperparameters, device)
    elif isinstance(init_strat, IcaInitStrategy):
        init_model_parameters_using_ica(model, X, n_components, scaled_hyperparameters, device, init_strat.iters)
    elif isinstance(init_strat, StandardInitStrategy):
        init_model_parameters_using_normal(model, scaled_hyperparameters, n_components)
    elif isinstance(init_strat, OnlyInitEncoderStrategy):
        print("will only initialise the encoder with normal distribution")
        init_encoder_using_normal(model, scaled_hyperparameters["sigma_enc"])
    return scaled_hyperparameters

# def train(
#     X,
#     n_components,
#     lr=1e-3,
#     epochs=2000,
#     batch_size=2048,
#     verbose=True,
#     init_strategy: InitStrategy=StandardInitStrategy(),
#     initialised_model=None,
#     use_ln_term=True,
#     sigma_s_rel_to_0="equal",
#     device="mps",
#     baseline_epochs=1000,
#     sigma_eps_override=None,
# ) -> SingleRun:
#     """
#     X: numpy array (n_samples, input_dim)
#     n_components: number of dictionary atoms
#     alpha: lorentzian sigma shrinker parameter
#     """
#     print("using hyperparameters")
#     print("\tinit_strategy", init_strategy)
#     print("\tuse_ln", use_ln_term)
#     print("\tsigma_s_rel_to_0", sigma_s_rel_to_0)
    
#     X_t = torch.tensor(X, dtype=torch.float32)
#     n_samples, input_dim = X_t.shape
#     X_t = X_t.to(device)

#     baseline_run = train_baseline(X, n_components, lr, baseline_epochs, batch_size, verbose)
#     sigma_eps = np.sqrt(baseline_run.loss)
#     tol = X.std() / 10_000
#     if sigma_eps < tol:
#         print(f"WARM: sigma_eps={sigma_eps} is less than tolerance={tol}, this can have undesired behavior")
#         sigma_eps = tol
#     if sigma_eps_override is not None:
#         sigma_eps = sigma_eps_override

#     print("baseline MSE", baseline_run.loss)

#     if initialised_model is None:
#         model = Autoencoder(input_dim, n_components)
#     else:
#         model = initialised_model

#     model = model.to(device)


#     scaled_hyperparameters = get_hyperparameters_and_init(
#         model, X, n_components, input_dim, sigma_eps, init_strategy, sigma_s_rel_to_0, device
#     )

#     p = scaled_hyperparameters
#     sigma_eps, sigma_0, sigma_s, _, alpha = (
#         p["sigma_eps"],
#         p["sigma_0"],
#         p["sigma_s"],
#         p["sigma_enc"],
#         p["alpha"],
#     )
#     print("using hyperparameters:", p)

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     if isinstance(init_strategy, WarmupInitStrategy):
#         warmup_with_l2(
#             X_t,
#             model,
#             init_strategy.warmup_epochs,
#             optimizer,
#             batch_size,
#             sigma_eps,
#             sigma_s,
#             sigma_0,
#             verbose=verbose,
#             device=device,
#         )


#     last_print_time = time.time()

#     for epoch in range(epochs):
#         # shuffle
#         idx = torch.randperm(n_samples, device=device)
#         permuted_X_t = X_t[idx]

#         for i in range(0, n_samples, batch_size):
#             batch = permuted_X_t[i : i + batch_size]
#             recon, codes = model(batch)

#             _recon_loss = recon_loss(batch, recon, sigma_eps)
#             _codes_loss = codes_loss(codes, sigma_s)
#             comp1, comp2 = weights_loss_cycled(alpha, sigma_0, model.decoder.weight)

#             if use_ln_term:
#                 weight_loss = comp1 + comp2
#             else:
#                 weight_loss = comp1

#             loss = _recon_loss + weight_loss + _codes_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if verbose and epoch % 200 == 0:
#             print(
#                 f"epoch {epoch:4d} | recon_loss {_recon_loss:.4f} weight_loss {weight_loss:.4f} codes_loss {_codes_loss:.4f} duration={time.time() - last_print_time}"
#             )
#             last_print_time = time.time()

#     with torch.no_grad():
#         recon, codes = model(X_t)

#     return SingleRun(
#         model.to("cpu"),
#         codes.to("cpu").numpy(),
#         model.decoder.weight.T.detach().to("cpu").numpy(),
#         recon.to("cpu").numpy(),
#         ((X_t - recon) ** 2).to("cpu").mean(),
#         scaled_hyperparameters,
#     )


def warmup_with_l2(
    X_t, model, epochs, optimizer, batch_size, sigma_eps, sigma_s, sigma_0, verbose=True, device="mps"
):
    print("starting warmup: epochs =", epochs)
    n_samples, input_dim = X_t.shape
    for epoch in range(epochs):
        idx = torch.randperm(n_samples, device=device)
        permuted_X_t = X_t[idx]
        for i in range(0, n_samples, batch_size):
            batch = permuted_X_t[i : i + batch_size]
            recon, codes, _ = model(batch)
            _recon_loss = recon_loss(batch, recon, sigma_eps)
            _codes_loss = codes_loss(codes, sigma_s)
            _weights_loss = gauss_loss(model.decoder.weight, 0) / (sigma_0 * sigma_0)

            loss = _recon_loss + _codes_loss + _weights_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and epoch % 200 == 0:
            print(f"warmup: epoch {epoch:4d} | recon_loss {_recon_loss:.4f}")




# def train_baseline(
#     X,
#     n_components,
#     lr=1e-3,
#     epochs=2000,
#     batch_size=256,
#     verbose=True,
# ):
#     """Train the autoencoder with just reconstruction loss, to find an arbitrary linear model which fits the data

#     The main training code requires sigma_eps
#     the standard deviation of expected gaussian noise
#     when the curve is fitted using Y=WX
#     We can generally do a simple sweep of hyperparams
#     or use simple heuristics
#     If the data is linearly "fittable",
#     then we get a good starting point
#     using this function.
#     """
#     print(f"training baseline model, epochs={epochs}")
#     X_t = torch.tensor(X, dtype=torch.float32)
#     n_samples, input_dim = X_t.shape

#     model = Autoencoder(input_dim, n_components)

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         idx = torch.randperm(n_samples)
#         permuted_X_t = X_t[idx]
#         for i in range(0, n_samples, batch_size):
#             batch = permuted_X_t[i : i + batch_size]
#             recon, codes = model(batch)
#             _recon_loss = recon_loss(batch, recon, 1)
#             loss = _recon_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         if verbose and epoch % 200 == 0:
#             print(f"finetune epoch {epoch:4d} | recon_loss {_recon_loss:.4f}")

#     with torch.no_grad():
#         recon, codes = model(X_t)

#     return SingleRun(
#         model,
#         codes.numpy(),
#         model.decoder.weight.T.detach().numpy(),
#         recon.numpy(),
#         ((X_t - recon) ** 2).mean().item(),
#         {},
#     )