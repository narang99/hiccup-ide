def get_recon_loss(x, recons, sigma_eps):
    """Reconstruction loss. MSE"""
    return get_gauss_loss(x, recons) / (sigma_eps * sigma_eps)


def get_codes_loss(codes, sigma_s):
    """L2 loss on encoder"""
    return get_gauss_loss(codes, 0) / (sigma_s * sigma_s)


def get_gauss_loss(x, mean):
    loss = (x - mean) ** 2
    # return torch.sum(loss, 1).mean()
    return loss.mean()
