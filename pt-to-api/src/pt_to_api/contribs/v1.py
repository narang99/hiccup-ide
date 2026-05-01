import torch.nn.functional as F
import itertools
import torch


def get_conv_input_indices_vectorized(kernel_size, stride, padding, out_h, out_w):
    """
    out_h, out_w: 1D tensors of shape [N] (integer indices)
    Returns starts, ends each of shape [N, 2] (row, col)
    """
    k, s, p = kernel_size, stride, padding

    in_r_start = out_h * s[0] - p[0]
    in_c_start = out_w * s[1] - p[1]

    # single-pixel case: out_coord_end = (out_h + 1, out_w + 1)
    in_r_end = out_h * s[0] + k[0] - p[0]
    in_c_end = out_w * s[1] + k[1] - p[1]

    starts = torch.stack([in_r_start, in_c_start], dim=1)  # [N, 2]
    ends   = torch.stack([in_r_end,   in_c_end  ], dim=1)  # [N, 2]
    return starts, ends

def get_conv_input_indices_for_patch_using_raw_params(
    kernel_size, stride, padding, out_coord_start, out_coord_end=None
):
    # Extracting parameters from the PyTorch layer
    k = kernel_size
    s = stride
    p = padding

    (r_start, c_start) = out_coord_start
    if out_coord_end is None:
        out_coord_end = r_start + 1, c_start + 1
    (r_end, c_end) = out_coord_end

    # Calculate input boundaries using the standard formula
    # Note: r_end and c_end are exclusive (standard slicing)
    in_r_start = r_start * s[0] - p[0]
    in_c_start = c_start * s[1] - p[1]

    # The end coordinate covers the full kernel of the last output pixel
    in_r_end = (r_end - 1) * s[0] + k[0] - p[0]
    in_c_end = (c_end - 1) * s[1] + k[1] - p[1]

    return (in_r_start, in_c_start), (in_r_end, in_c_end)


def get_conv_input_indices_for_patch(layer, out_coord_start, out_coord_end=None):
    # Extracting parameters from the PyTorch layer
    return get_conv_input_indices_for_patch_using_raw_params(
        layer.kernel_size, layer.stride, layer.padding, out_coord_start, out_coord_end
    )


def _get_input_patch_indices_for_point_assuming_padded_input(layer, out_y, out_x):
    """
    Returns the (y_start, y_end, x_start, x_end) indices of the input
    tensor used to calculate the output pixel at (out_y, out_x).
    """
    # Extract parameters from the PyTorch submodule
    # we handle padding by actually padding the input
    # so the convolution works out fine
    # the coordinates we return assume that it is already padded and so are relative to the padded tensor
    # if you want to send the user which coordinates were actually used
    # you should subtract the padding
    # _get_actual_pad_coords is useful here
    # note that im not handling reflection padding here
    # technically, we should simply show the bigger area for our convolutions contributions
    # since padded convolutions still contribute (unless they are zero)
    # for now, lite lera since keeping track across layers can be slightly hard
    k = layer.kernel_size
    s = layer.stride

    # Calculate top-left corner
    y_start = out_y * s[0]
    x_start = out_x * s[1]

    # Calculate bottom-right corner
    y_end = y_start + k[0]
    x_end = x_start + k[1]

    # Note: Indices might be negative if they fall into the padding zone
    return (y_start, y_end, x_start, x_end)


def get_padded_for_conv(layer, tens):
    padding_mode = layer.padding_mode
    if layer.padding_mode == "zeros":
        padding_mode = "constant"
    padded = F.pad(tens, layer._reversed_padding_repeated_twice, padding_mode)
    return padded


def conv_calculate_raw_contribs_for_single_out_pixel(
    layer, acts_layer_in, b, k, out_y, out_x
):
    # we return the patch relative to the padded input itself
    padded = get_padded_for_conv(layer, acts_layer_in)

    y0, y1, x0, x1 = _get_input_patch_indices_for_point_assuming_padded_input(
        layer, out_y, out_x
    )
    # we do pointwise mult for this one
    cube = padded[b, :, y0:y1, x0:x1].cpu()
    kernel = layer.weight[k].cpu()
    bias = layer.bias[k].cpu()
    pointwise = cube * kernel
    bias_for_each_component = bias / pointwise.numel()

    # this is for checking
    # verified
    calculated_activation = pointwise.sum() + bias

    contribs = pointwise + bias_for_each_component
    total_sum = pointwise.abs().sum() + bias.abs()
    contribs /= total_sum
    # we need to return the patch also
    return (
        contribs,
        calculated_activation,
        (y0, y1, x0, x1),
    )


def verify_manual_conv_works(layer, acts_layer_in, acts_layer_out):
    # go through all output activations
    # for each, find the input patch
    # for each find the
    res = torch.zeros(acts_layer_out.shape).to(torch.float32)

    for b in range(len(acts_layer_out)):
        for k in range(len(acts_layer_out[b])):
            for i in range(len(acts_layer_out[b][k])):
                for j in range(len(acts_layer_out[b][k][i])):
                    _, calculated_activation, _ = (
                        conv_calculate_raw_contribs_for_single_out_pixel(
                            layer, acts_layer_in, b, k, i, j
                        )
                    )
                    res[b][k][i][j] = calculated_activation
    assert torch.allclose(acts_layer_out, res, atol=1e-6)
    print("All good ✅")


def conv_calculate_contribs_for_all(
    layer, acts_layer_in, acts_layer_out, out_contribs, k=None
):
    # if you want the contribs of a specific kernel, pass `k` as a non null value
    all_contribs = get_padded_for_conv(
        layer, torch.zeros(acts_layer_in.shape).to(torch.float32)
    )

    for b in range(len(acts_layer_out)):
        k_range = range(len(acts_layer_out[b]))
        if k is not None:
            # constrain the k if the user asked
            k_range = range(k, k + 1)
        for k in k_range:
            for i in range(len(acts_layer_out[b][k])):
                for j in range(len(acts_layer_out[b][k][i])):
                    # these are the contribs for all the input patch
                    in_contribs, calced_val, patch_coords = (
                        conv_calculate_raw_contribs_for_single_out_pixel(
                            layer, acts_layer_in, b, k, i, j
                        )
                    )
                    y0, y1, x0, x1 = patch_coords
                    # we use the same cube we used before the setting the contribs
                    # we also multiply the contribs with the actual contrib of the output neuron
                    all_contribs[b][:, y0:y1, x0:x1] += (
                        in_contribs * out_contribs[b][k][i][j]
                    )
                    if not torch.allclose(
                        calced_val, acts_layer_out[b][k][i][j], atol=1e-6
                    ):
                        print(
                            "dtypes", calced_val.dtype, acts_layer_out[b][k][i][j].dtype
                        )
                        raise Exception(
                            f"found manually calculated convolution result to be different than actual activation.\nCalculated={calced_val.item()} Actual={acts_layer_out[b][k][i][j].item()}"
                        )

    pady, padx = layer.padding
    return all_contribs[:, :, pady:-pady, padx:-padx]


def linear_calculate_contribs_for_all(
    layer, acts_layer_in, acts_layer_out, out_contribs
):
    weight, bias = layer.weight, layer.bias
    all_contribs = torch.zeros(acts_layer_in.shape).to(torch.float32)

    for b in range(len(acts_layer_out)):
        for out_act_i in range(len(acts_layer_out[b])):
            pointwise = weight[out_act_i] * acts_layer_in[b]
            this_bias = bias[out_act_i]
            bias_for_each_component = this_bias / pointwise.numel()
            contribs = pointwise + bias_for_each_component
            total_sum = pointwise.abs().sum() + this_bias.abs()
            contribs /= total_sum

            all_contribs[b] += contribs * out_contribs[b][out_act_i]

    return all_contribs


def relu_calculate_contribs(out_contribs):
    # for now, we say that ReLU does not change contribs, its just an ID function, contribs flow back as is
    return out_contribs


def do_slicewise_convolutions(in_acts, kernels, stride, padding, dilation):
    # in acts would be of size (<chan>, h, w). kernels are of size (<chan>, k, k)
    slice_convs = []
    for act, kernel in zip(in_acts, kernels):
        conv_res = F.conv2d(
            act[None, None, ...],
            kernel[None, None, ...],
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        slice_convs.append(conv_res[0][0])
    return slice_convs

def calc_contribs_for_slices(slice_convs, bias, out_contrib):
    h, w = slice_convs[0].shape
    # slice_convs = torch.stack(slice_convs)
    slice_contribs = torch.zeros(slice_convs.shape)
    each_elements_bias = bias / slice_convs.shape[0]

    for i, j in itertools.product(range(h), range(w)):
        total_sum = slice_convs[:, i, j].abs().sum()
        total_sum += bias.abs()

        for d in range(slice_convs.shape[0]):
            slice_contribs[d, i, j] = (
                slice_convs[d, i, j] + each_elements_bias
            ) / total_sum
            slice_contribs[d, i, j] *= out_contrib[i, j]
    return slice_contribs

def decompose_3d_contrib_to_slicewise_contribs(
    acts, kernel, bias, out_contrib, stride, padding, dilation
):
    slice_convs = do_slicewise_convolutions(acts, kernel, stride, padding, dilation)
    slice_convs = torch.stack(slice_convs)
    slice_contribs = calc_contribs_for_slices(slice_convs, bias, out_contrib)
    return slice_convs, slice_contribs
