import matplotlib.pyplot as plt


def save_combined_scatter_png(activations, noise_samples, output_path):
    """
    Saves a single dark-theme-friendly scatter plot with both series overlaid:
    noise samples in a faded gray, activation values in red — each plotted
    against its own 0..len(series)-1 x-index (matching the
    axes[1].scatter(range(len(match.noise)), match.noise) idiom used in
    NeuronParentAnalyser.plot_clusters), so the two series aren't forced onto a
    shared x-axis meaning. Headless (savefig, no plt.show) since this is always
    regenerated during report building rather than viewed inline in a notebook.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#212529")
    ax.set_facecolor("#212529")
    if len(noise_samples) > 0:
        ax.scatter(
            range(len(noise_samples)), noise_samples, s=12, color="#888888", label="noise"
        )
    ax.scatter(
        range(len(activations)), activations, s=12, color="red", label="output_activation"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    legend = ax.legend(facecolor="#212529", labelcolor="white")
    legend.get_frame().set_edgecolor("white")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
