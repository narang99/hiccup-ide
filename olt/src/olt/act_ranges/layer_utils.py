def get_layer_params(model, current_layer_name, current_channel):
    layer = model.get_submodule(current_layer_name)
    padding = layer.padding
    ksize, stride = layer.kernel_size, layer.stride
    w = layer.weight[current_channel].detach().cpu()
    return w, ksize, stride, padding


class ReceptiveFieldOutOfBounds(ValueError):
    """Raised by receptive_block when a computed receptive field falls outside
    its tensor's valid range. A dedicated type (rather than a bare ValueError)
    so callers that recover from this (skip the affected probe/firing/sample —
    see analyser.top_contributing_indices, similarity.get_neuron_closest_cluster,
    feature_viz.dump_feature_viz_asset) catch exactly this condition, not some
    unrelated ValueError raised nearby."""


def receptive_block(i, ksize, stride, padding, input_size=None):
    """
    Returns [start, end) input indices (end=exclusive) that influence
    output position i of a conv layer.

    If input_size is given, raises ReceptiveFieldOutOfBounds when the computed
    range falls outside [0, input_size) instead of clamping it. This module
    always multiplies the sliced patch elementwise against a fixed-shape conv
    weight (see analyser.top_contributing_indices, similarity.get_neuron_closest_cluster),
    so a clamped/narrower-than-ksize patch would just fail later with a shape
    mismatch anyway — and a negative start left unclamped would silently wrap
    around via Python/numpy/torch's negative-index slicing, pulling data from
    the wrong end of the tensor instead of erroring. Every caller that's
    slicing a real tensor should pass input_size (from that tensor's own
    shape) so an out-of-range case fails immediately at the coordinate-math
    step, not later as a confusing shape error or, worse, not at all. This
    currently never fires for the one supported current_layer (a
    1x1/stride-1/pad-0 conv, so y0/x0 can never go negative — see
    olt/CLAUDE.md) — it's here so extending to a new layer with a real kernel
    fails fast instead of silently, and so callers can recover per-atom (see
    ReceptiveFieldOutOfBounds) instead of the whole run dying on one
    boundary position.
    """
    start = i * stride - padding
    end = start + ksize

    if input_size is not None and (start < 0 or end > input_size):
        raise ReceptiveFieldOutOfBounds(
            f"receptive_block: computed range [{start}, {end}) for output index "
            f"{i} (ksize={ksize}, stride={stride}, padding={padding}) falls "
            f"outside the input's valid range [0, {input_size}) — this usually "
            "means a current_layer/dep_layer this module wasn't designed for "
            "(see olt/CLAUDE.md)."
        )

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
