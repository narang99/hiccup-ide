import torch
import torch.nn.functional as F


def linear_calculate_contribs_for_all(layer, acts_layer_in, out_contribs, device="cpu"):
    weight, bias = layer.weight, layer.bias

    pointwise = acts_layer_in.unsqueeze(1) * weight.unsqueeze(0)
    num_ele_in_each_pw = torch.tensor(pointwise[0][0].numel()).to(device)
    bias_tens = bias.unsqueeze(0).unsqueeze(-1) / num_ele_in_each_pw

    contribs = pointwise + bias_tens
    total_sum = pointwise.abs().sum(dim=-1) + bias.abs()
    total_sum = total_sum.unsqueeze(-1)
    contribs /= total_sum

    scaled_contribs = (contribs * out_contribs.unsqueeze(-1)).sum(dim=1)
    return scaled_contribs


def do_slicewise_convolutions(in_acts, weight, stride, padding, dilation):
    # in_acts shape: [N, 8, 14, 14]
    # weight shape:  [16, 8, 3, 3]
    N, C_in, H, W = in_acts.shape
    C_out, _, K_h, K_w = weight.shape

    # 1. Reshape weights to treat every slice as an independent filter

    # layout of weight flat
    # [cin=0, cout=0], [cin=0, cout=1] ... [cin=0, cout=C_out]
    # [cin=1, cout=0], [cin=1, cout=1] ... [cin=1, cout=C_out]
    # .. so on
    weight_flat = (
        weight.permute(1, 0, 2, 3).contiguous().view(C_in * C_out, 1, K_h, K_w)
    )

    # 2. Expand input so each kernel's slice has its own input channel
    # We repeat the channels: [Ch1, Ch2...] -> [Ch1, Ch1...(16 times), Ch2, Ch2...]
    # Using repeat_interleave ensures alignment with our flattened weights
    # New shape: [N, 128, 14, 14]
    # layout of acts
    # same as weight_flat
    # [cin=0, cout=0], [cin=0, cout=1] ... [cin=0, cout=C_out]
    # [cin=1, cout=0], [cin=1, cout=1] ... [cin=1, cout=C_out]
    # .. so on
    in_acts_expanded = in_acts.repeat_interleave(C_out, dim=1)

    # 3. Execute all 128 2D convolutions in one GPU call
    # Each group handles exactly 1 input channel and 1 filter slice
    output = F.conv2d(
        in_acts_expanded,
        weight_flat,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=C_in * C_out,
    )

    # 4. Reshape back into a readable "Cube of Cubes"
    # Result: [N, C_in, C_out, H_out, W_out]
    # (or [N, 8, 16, 7, 7])
    return output.view(N, C_in, C_out, output.shape[-2], output.shape[-1])


def get_slice_contribs(slice_convs, bias, out_contribs):
    total_sum = slice_convs.abs().sum(dim=1) + bias.view(1, -1, 1, 1).abs()
    total_sum = total_sum.unsqueeze(1)
    total_slices = slice_convs.shape[1]
    each_elements_bias = bias.view(1, 1, -1, 1, 1) / total_slices

    slice_contribs = (slice_convs + each_elements_bias) / total_sum
    slice_contribs = slice_contribs * out_contribs.unsqueeze(1)

    return slice_contribs


def conv_calculate_contribs_for_all(in_acts, layer, out_contribs):
    # the whole thing depthwise is this only, now i get why they multiply the in_chan with kk
    # cuz thats the normal operation lol
    chan_in = in_acts.shape[1]
    out_h, out_w = out_contribs.shape[-2:]
    in_h, in_w = in_acts.shape[-2:]

    # [b, n_in * k * k, L]
    unfolded = F.unfold(
        in_acts, layer.kernel_size, layer.dilation, layer.padding, layer.stride
    )

    # [n_out, n_in, k, k] -> [n_out, n_in*k*k]
    weight = layer.weight.flatten(start_dim=-3)
    # [1, n_out, n_in*k*k, 1]
    weight = weight.unsqueeze(0).unsqueeze(-1)
    # [b, 1, n_in*k*k, L] (aligned with weight)
    unfolded = unfolded.unsqueeze(1)

    # [1, n_out, 1, 1] (aligned with weight and unfolded)
    bias = layer.bias.view(1, -1, 1, 1)

    # [b, n_out, n_in*k*k, L]
    pointwise = weight * unfolded
    # [b, n_out, 1, L]
    abs_sum = pointwise.abs().sum(dim=-2, keepdim=True)
    abs_sum += bias.abs()

    pointwise += bias / pointwise.shape[-2]

    # [b, n_out, n_in*k*k, L]
    contribs = pointwise / abs_sum

    # collapse the chan_out in batch dim
    s = contribs.shape
    # [b, n_out, out_h, out_w] -> [b, n_out, 1, L] (aligned with contribs)
    our_out_contribs = out_contribs.flatten(start_dim=-2).unsqueeze(-2)
    contribs *= our_out_contribs

    cs = contribs.shape
    # [b, chan_out, chan_in, k*k, out_h, out_w] -> [b, chan_out, chan_in, out_h, out_w]
    slice_contribs = contribs.reshape(
        cs[0], cs[1], chan_in, -1, out_h, out_w
    ).sum(dim=-3)

    contribs = contribs.reshape(s[0] * s[1], s[2], s[3])
    folded = F.fold(
        contribs,
        (in_h, in_w),
        layer.kernel_size,
        layer.dilation,
        layer.padding,
        layer.stride,
    )
    contribs = folded.reshape(s[0], s[1], chan_in, in_h, in_w).sum(dim=1)
    return contribs, slice_contribs


def conv_calculate_contribs_through_slices(in_acts, layer, out_contribs):
    # a different algorithm, not consistent with v1 def
    # we first find slice wise contribs, then flow to input
    # note that our calculations do not align even though logically they should
    # this is because we are dividing by absolute values
    # another way can be to calculate slice wise contribs from the raw contribs themselves
    slice_convs = do_slicewise_convolutions(
        in_acts, layer.weight, layer.stride, layer.padding, layer.dilation
    )
    slice_contribs = get_slice_contribs(slice_convs, layer.bias, out_contribs)
    conv_contribs = _get_conv_input_channel_contribs_from_slice_contribs(
        in_acts, slice_contribs, layer
    )
    return conv_contribs


def _get_conv_input_channel_contribs_from_slice_contribs(
    input_batch, slice_contribs, layer
):
    tol = torch.tensor(1e-9)
    # old: [b, in_chan, out_chan,  out_h, out_w]
    # new: [b, out_chan, in_chan, out_h, out_w]
    slice_contribs = slice_contribs.permute(0, 2, 1, 3, 4)

    # shape: (out_chan, in_chan, ksize, ksize)
    weight = layer.weight

    # shape: (out_chan, in_chan, ksize*ksize)
    reshaped_weights = weight.flatten(start_dim=-2)

    # ksize * ksize value
    # kernel_flat_weight_size = reshaped_weights.shape[-1]

    # shape (1, out_chan, in_chan, ksize*ksize, 1)
    reshaped_weights = reshaped_weights.unsqueeze(-1).unsqueeze(0)

    # input batch: [b, in_chan, inp_h, inp_w]
    inp_h, inp_w = input_batch.shape[-2:]
    out_chan, in_chan = layer.weight.shape[0], layer.weight.shape[1]

    # [b, in_chan * ksize * ksize, L]
    # L = out_h * out_w
    unfolded = F.unfold(
        input_batch, layer.kernel_size, layer.dilation, layer.padding, layer.stride
    )

    # [b, 1, in_chan, ksize * ksize, L]
    # now this is aligned with reshaped_weights
    unfolded = unfolded.reshape(
        unfolded.shape[0], in_chan, -1, unfolded.shape[-1]
    ).unsqueeze(1)

    # multiply and scale
    # [b, out_chan, in_chan, ksize*ksize, L]
    pointwise = unfolded * reshaped_weights

    # sum done across ksize*ksize, so along each patch, [b, out_chan, in_chan, 1, L]
    sum_for_each_patch = pointwise.abs().sum(dim=-2, keepdim=True)

    # bias shape: from=[out_chan] to=[1, out_chan, 1, 1, 1]
    bias = layer.bias.view(1, -1, 1, 1, 1)
    # since we are only doing for each slice, we divide the bias by the number of in_chan
    bias = bias / in_chan

    # add bias to total sum for each patch (we spread the bias across slices)
    sum_for_each_patch += bias.abs()
    # prevent 0 in sum, we dont wanna divide by 0
    sum_for_each_patch[sum_for_each_patch < tol] = tol

    # spread bias in pointwise
    pointwise += bias / pointwise.shape[-2]

    # [b, out_chan, in_chan, ksize*ksize, L]
    pointwise = pointwise / sum_for_each_patch

    # [b, out_chan, in_chan, 1, L]
    reshaped_slice_contribs = slice_contribs.flatten(start_dim=-2).unsqueeze(-2)
    scaled_with_contrib_pointwise = reshaped_slice_contribs * pointwise

    N, out_C, in_C, kk, L = scaled_with_contrib_pointwise.shape
    # collapse out_c in the batch dim since fold only takes 3 dims
    to_fold = scaled_with_contrib_pointwise.reshape(N * out_C, in_C * kk, L)
    folded = F.fold(
        to_fold,
        (inp_h, inp_w),
        layer.kernel_size,
        layer.dilation,
        layer.padding,
        layer.stride,
    )

    return folded.reshape(N, out_C, in_C, inp_h, inp_w).sum(dim=1)
