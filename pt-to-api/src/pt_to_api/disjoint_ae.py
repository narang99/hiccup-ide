import torch
import numpy as np
from torch import nn
from torch import optim


def _get_weights_loss_on_decoder(model, alpha, sigma_0, weights_algo):
    if weights_algo == "cycled":
        weight_loss = model.weights_loss_cycled(alpha, sigma_0, model.decoder.weight)
    else:
        comp1, comp2 = model.weights_loss(alpha, sigma_0, model.decoder.weight)
        weight_loss = (comp1 + comp2).sum()
    return weight_loss

def get_alpha(epoch, total_epochs, alpha_start=0.0, alpha_end=1.0):
    # do the last 25% with max_alpha
    epoch_max = int(total_epochs * 0.30)
    return alpha_start + (alpha_end - alpha_start) * (epoch / epoch_max)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, n_components, init_sigma_eps=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_components, bias=True),
        )
        self.decoder = nn.Linear(n_components, input_dim, bias=False)

    def forward(self, x):
        coefficients = self.encoder(x)
        recon = self.decoder(coefficients)
        return recon, coefficients

    def recon_loss(self, x, recons, sigma_x):
        """Reconstruction loss. MSE"""
        return self.gauss_loss(x, recons) / (sigma_x * sigma_x)

    def coefficients_loss(self, sigma_s):
        """L2 loss on encoder"""
        return self.gauss_loss(self.encoder[0].weight, 0) / (sigma_s * sigma_s)

    def gauss_loss(self, x, mean):
        loss = (x - mean) ** 2
        return torch.sum(loss, 1).mean()

    def weights_loss_cycled(self, alpha, sigma_0, W):
        """
        This is similar to weight_loss function
        We basically run `weight_loss` starting from every dimension c, and then average them out.
        Useful for testing if there is a bias in the main `weights_loss` function
        """
        K = W.shape[1]
        shift_losses = []
        for s in range(K):
            comp1, comp2 = self.weights_loss(alpha, sigma_0, torch.roll(W, -s, dims=1))
            shift_losses.append((comp1 + comp2).sum())
        return torch.mean(torch.stack(shift_losses))

    def weights_loss(self, alpha, sigma_0, W):
        """Vectorized version the Weight loss"""
        W_sq = W**2  # (C, K)
        cumsum = torch.cumsum(W_sq, dim=1)  # (C, K), cumsum[c,k] = sum W[c,0..k]^2
        phi = alpha * torch.roll(cumsum, 1, dims=1) + 1  # (C, K)
        phi[:, 0] = 1  # k=0: phi_weight(W, c, -1, alpha) = alpha*0 + 1
        comp1 = (W_sq * phi) / (sigma_0 * sigma_0)
        comp2 = -torch.log(phi)
        return comp1, comp2

class LearnedSigmaAutoencoder(nn.Module):
    def __init__(self, input_dim, n_components, init_sigma_eps=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_components, bias=True),
        )
        self.decoder = nn.Linear(n_components, input_dim, bias=False)
        eps_tol = 1e-5
        sigma_eps = init_sigma_eps if init_sigma_eps is not None else eps_tol
        self.sigma_eps = nn.Parameter(torch.tensor(sigma_eps))
        # self.sigma_0 = self.sigma_eps * 5
        # self.sigma_s = self.sigma_0 * 10

    def forward(self, x):
        coefficients = self.encoder(x)
        recon = self.decoder(coefficients)
        return recon, coefficients

    def recon_loss(self, x, recons, sigma_s):
        """Reconstruction loss. MSE"""
        return self.gauss_loss(x, recons) / (self.sigma_eps * self.sigma_eps)

    def coefficients_loss(self, sigma_s):
        """L2 loss on encoder"""
        return self.gauss_loss(self.encoder[0].weight, 0) / (self.sigma_eps * self.sigma_eps * 50 * 50)

    def gauss_loss(self, x, mean):
        loss = (x - mean) ** 2
        return torch.sum(loss, 1).mean()

    def weights_loss_cycled(self, alpha, sigma_0, W):
        """
        This is similar to weight_loss function
        We basically run `weight_loss` starting from every dimension c, and then average them out.
        Useful for testing if there is a bias in the main `weights_loss` function
        """
        K = W.shape[1]
        shift_losses = []
        for s in range(K):
            comp1, comp2 = self.weights_loss(alpha, sigma_0, torch.roll(W, -s, dims=1))
            shift_losses.append((comp1 + comp2).sum())
        return torch.mean(torch.stack(shift_losses))

    def weights_loss(self, alpha, sigma_0, W):
        """Vectorized version the Weight loss"""
        W_sq = W**2  # (C, K)
        cumsum = torch.cumsum(W_sq, dim=1)  # (C, K), cumsum[c,k] = sum W[c,0..k]^2
        phi = alpha * torch.roll(cumsum, 1, dims=1) + 1  # (C, K)
        phi[:, 0] = 1  # k=0: phi_weight(W, c, -1, alpha) = alpha*0 + 1
        comp1 = (W_sq * phi) / (self.sigma_eps * self.sigma_eps * 5 * 5)
        comp2 = -torch.log(phi)
        return comp1, comp2


def train(
    X,
    n_components,
    alpha=5000,
    sigma_eps=0.1,
    sigma_s=1,
    sigma_0=1,
    lr=1e-3,
    epochs=2000,
    batch_size=256,
    weights_algo="cycle",
    verbose=True,
    svd_init=False,
):
    """
    X: numpy array (n_samples, input_dim)
    n_components: number of dictionary atoms
    alpha: lorentzian sigma shrinker parameter
    sigma_eps: std of noise in the data after modelling the data as a W@S
    sigma_0: std of W, useful to keep very near 0
    sigma_s: sigma for the gaussian distribution for sampling the encoder weights. It indirectly restricts its outputs (the coefficients) to be gaussian
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    n_samples, input_dim = X_t.shape

    model = LearnedSigmaAutoencoder(input_dim, n_components)

    if svd_init:
        print("using svd init for decoder")
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        model.decoder.weight.data = torch.tensor(Vt[:n_components].T, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        # shuffle
        idx = torch.randperm(n_samples)
        permuted_X_t = X_t[idx]

        for i in range(0, n_samples, batch_size):
            batch = permuted_X_t[i : i + batch_size]
            recon, codes = model(batch)

            recon_loss = model.recon_loss(batch, recon, sigma_eps)
            coefficients_loss = model.coefficients_loss(sigma_s)
            weight_loss = _get_weights_loss_on_decoder(
                model, get_alpha(epoch, epochs, 100, alpha), sigma_0, weights_algo
            )
            loss = recon_loss + weight_loss + coefficients_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        if verbose and epoch % 200 == 0:
            print(
                f"epoch {epoch:4d} | recon_loss {recon_loss:.4f} weight_loss {weight_loss.sum():.4f} coefficients_loss {coefficients_loss:.4f}"
            )

    with torch.no_grad():
        recon, codes = model(X_t)
    return (
        model,
        codes.numpy(),
        model.decoder.weight.T.detach().numpy(),
        recon.numpy(),
    )


def train_baseline(
    X, n_components, lr=1e-3, epochs=2000, batch_size=256, sigma_x=1, verbose=True
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
    X_t = torch.tensor(X, dtype=torch.float32)
    n_samples, input_dim = X_t.shape

    model = Autoencoder(input_dim, n_components)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    model.decoder.weight.data = torch.tensor(Vt[:n_components].T, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)
        permuted_X_t = X_t[idx]
        for i in range(0, n_samples, batch_size):
            batch = permuted_X_t[i : i + batch_size]
            recon, codes = model(batch)
            recon_loss = model.recon_loss(batch, recon, sigma_x)
            loss = recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and epoch % 200 == 0:
            print(f"finetune epoch {epoch:4d} | recon_loss {recon_loss:.4f}")

    with torch.no_grad():
        recon, codes = model(X_t)

    return (
        model,
        codes.numpy(),
        model.decoder.weight.T.detach().numpy(),  # (n_components, input_dim)
        recon.numpy(),
    )
