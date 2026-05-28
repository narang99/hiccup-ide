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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, n_components, initial_beta_eps=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_components, bias=True),
        )
        self.decoder = nn.Linear(n_components, input_dim, bias=False)
        self.beta_0 = nn.Parameter(torch.tensor(0.)).float()
        self.beta_0.requires_grad_()
        init_val = 0. if initial_beta_eps is None else initial_beta_eps
        self.beta_eps = nn.Parameter(torch.tensor(init_val)).float()
        print("using init val", self.beta_eps)
        self.beta_eps.requires_grad_()
        self.beta_s = nn.Parameter(torch.tensor(0.)).float()
        self.beta_s.requires_grad_()

    def forward(self, x):
        coefficients = self.encoder(x)
        recon = self.decoder(coefficients)
        return recon, coefficients

    def recon_loss(self, x, recons):
        """Reconstruction loss. MSE"""
        return self.gauss_loss(x, recons, self.beta_eps)

    def coefficients_loss(self):
        """L2 loss on encoder"""
        return self.gauss_loss(self.encoder[0].weight, 0, self.beta_s)

    def gauss_loss(self, x, mean, beta):
        loss = (x - mean) ** 2
        return torch.sum(loss, 1).mean()*torch.exp(-beta) + beta

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

    def weights_loss(self, alpha, W):
        """Vectorized version the Weight loss"""
        W_sq = W**2  # (C, K)
        cumsum = torch.cumsum(W_sq, dim=1)  # (C, K), cumsum[c,k] = sum W[c,0..k]^2
        phi = 1 + alpha * torch.exp(-self.beta_0) * torch.roll(cumsum, 1, dims=1)  # (C, K)
        phi[:, 0] = 1  # k=0: phi_weight(W, c, -1, alpha) = alpha*0 + 1
        comp1 = (W_sq * phi * torch.exp(-self.beta_0))
        comp2 = -torch.log(phi)
        comp3 = self.beta_0
        return (comp1 + comp2 + comp3).sum()


def train(
    X,
    n_components,
    alpha=5000,
    lr=1e-3,
    epochs=2000,
    batch_size=256,
    verbose=True,
    svd_init=False,
    initial_beta_eps=None,
):
    """
    X: numpy array (n_samples, input_dim)
    n_components: number of dictionary atoms
    alpha: lorentzian sigma shrinker parameter
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    n_samples, input_dim = X_t.shape

    model = Autoencoder(input_dim, n_components, initial_beta_eps)

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

            # print("passing batch", batch.shape)
            recon, codes = model(batch)

            recon_loss = model.recon_loss(batch, recon)
            coefficients_loss = model.coefficients_loss()
            weight_loss = model.weights_loss(alpha, model.decoder.weight)

            loss = recon_loss + weight_loss + coefficients_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 200 == 0:
            print(
                f"epoch {epoch:4d} | recon_loss {recon_loss:.4f} weight_loss {weight_loss:.4f} coefficients_loss {coefficients_loss:.4f}"
            )

    with torch.no_grad():
        recon, codes = model(X_t)
    return (
        model,
        codes.numpy(),
        model.decoder.weight.T.detach().numpy(),
        recon.numpy(),
    )
