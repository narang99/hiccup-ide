import numpy as np


def get_inferred_sigma_eps_if_needed(
    X,
    n_components,
    lr,
    baseline_epochs,
    batch_size,
    sigma_eps_override,
    verbose,
    train_baseline_fn,
):
    """Framework-agnostic sigma_eps inference that works with any train_baseline function."""
    if sigma_eps_override is not None:
        print("sigma_eps_override passed, skipping baseline run")
        sigma_eps = sigma_eps_override
        baseline_loss = None
    else:
        baseline_run = train_baseline_fn(
            X=X,
            n_components=n_components,
            lr=lr,
            epochs=baseline_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        sigma_eps = np.sqrt(baseline_run.loss)
        tol = X.std() / 10_000
        if sigma_eps < tol:
            print(
                f"WARM: sigma_eps={sigma_eps} is less than tolerance={tol}, this can have undesired behavior"
            )
            sigma_eps = tol
        print("baseline MSE", baseline_run.loss)
        baseline_loss = baseline_run.loss
    return sigma_eps, baseline_loss


def get_scaled_hyperparamters(
    sigma_x,
    input_dim,
    n_components,
    sigma_eps,
    w_to_eps_ratio=5,
    alpha_constant=5000.0,
    sigma_s_rel_to_0="equal",
):
    """
    sigma_x      : std of your data
    eps_ratio    : sigma_x / sigma_eps (default 100)
    w_to_eps_ratio: sigma_0 / sigma_eps (default 5)
    alpha_constant: the c in alpha = c / sigma_0^2
    """
    if sigma_s_rel_to_0 == "equal":
        # sigma_0 = sigma_s = sigma_x / np.sqrt(n_components)
        sigma_0 = sigma_s = 1
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
