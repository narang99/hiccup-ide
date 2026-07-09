def get_layer_params(model, current_layer_name, current_channel):
    layer = model.get_submodule(current_layer_name)
    padding = layer.padding
    ksize, stride = layer.kernel_size, layer.stride
    w = layer.weight[current_channel].detach().cpu()
    return w, ksize, stride, padding


def receptive_block(i, ksize, stride, padding, input_size=None):
    """
    Returns [start, end) input indices (end=exclusive) that influence
    output position i of a conv layer.
    """
    start = i * stride - padding
    end = start + ksize

    if input_size is not None:
        start = max(start, 0)
        end = min(end, input_size)

    return start, end


class FlattenedChannelMap:
    """
    Maps a flattened channel index in a concatenated dependency-block tensor
    back to the (layer_name, channel) that produced it, given an ordered
    branch layout (see constants/branches.py, e.g. MIXED4D_BRANCHES) mirroring
    the torch.cat(...) order in the model's forward().
    """

    def __init__(self, branches):
        self.branches = branches

    def get_layer_and_chan_from_flattened(self, flattened_chan):
        offset = 0
        for layer_name, n_channels in self.branches:
            if flattened_chan < offset + n_channels:
                return layer_name, flattened_chan - offset
            offset += n_channels
        raise ValueError(
            f"flattened_chan {flattened_chan} is out of range for branches {self.branches}"
        )
