def show_single_example(
    prev_contribs,
    prev_acts,
    contribs,
    pos_mask,
    neg_mask,
    layer_name,
    model,
    channel_by_code_maxes,
    channel_by_model,
):
    S([prev_acts[0][0], prev_contribs[0][0]], (4, 2), 2, viztype="local")
    plt.show()

    trimmed_contrib = get_trimmed_contrib(contribs, pos_mask, neg_mask)
    S([trimmed_contrib[0].reshape(trimmed_contrib.shape[1], -1)], (20, 5))
    plt.show()

    layer = model.get_submodule(layer_name)
    vals = do_for_one_input(
        trimmed_contrib, prev_acts, layer, channel_by_code_maxes, channel_by_model
    )
    S([vals], 20)
    plt.show()


def show_batch(
    contribs,
    acts,
    indices_to_show,
    pos_mask,
    neg_mask,
    layer_name,
    prev_layer_name,
    model,
    channel_by_code_maxes,
    channel_by_model,
):
    for i in indices_to_show:
        show_single_example(
            contribs[prev_layer_name][i][None, ...],
            acts[prev_layer_name][i][None, ...],
            contribs[layer_name][i][None, ...],
            pos_mask,
            neg_mask,
            layer_name,
            model,
            channel_by_code_maxes,
            channel_by_model,
        )
