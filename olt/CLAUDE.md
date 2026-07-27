# olt

## `act_ranges` — NeuronParentAnalyser and Inception coordinate mapping

`NeuronParentAnalyser` (`src/olt/act_ranges/analyser.py`) traces which dependency-layer
activations explain a given neuron's activation, for GoogLeNet/InceptionV1 (lucent's
`InceptionV1`). It currently only supports `current_layer_name="mixed4e_1x1_pre_relu_conv"`.

**Why only one layer is supported today**: `dep_coords` translates a receptive-field
coordinate in `current_layer`'s captured (post-pad) input tensor back to the dependency
block's raw (pre-pad) output coordinate, using a single manual pad offset
(`f_pad_manual_before_between_us_and_dep`, and its reference values in
`src/olt/act_ranges/constants/paddings.py`). This only works when `current_layer`'s conv
reads the dependency's concatenated output either directly (no pad) or through exactly one
manual `F.pad` — not through any additional op (e.g. a pooling layer) that has its own
receptive field. See `InceptionV1.forward` (in the `lucent` package) for the actual pad/pool/conv
sequence per mixed block before adding support for a new layer.

`mixed4e_pool_reduce_pre_relu_conv` is a known example that breaks this: `mixed4d` is
`F.pad`'d, then passed through a 3x3 maxpool (stride 1), and only then convolved — the
maxpool's own receptive field isn't accounted for by the single-offset logic. It's listed in
`UNSUPPORTED_CURRENT_LAYERS` (`constants/paddings.py`), and `NeuronParentAnalyser` raises if
passed as `current_layer_name`.

**When adding support for a new `current_layer_name`:**
1. Trace its path back to the dependency block's `torch.cat(...)` in `InceptionV1.forward`.
2. If the conv reads the concatenated tensor directly, or through exactly one `F.pad`, add
   the pad amount (or `(0, 0)`) to `F_PAD_MANUAL_BY_CURRENT_LAYER`.
3. If there's any other op (pooling, another conv, etc.) between the pad and the conv,
   `dep_coords`'s single-offset logic is not sufficient — extend it, and remove the layer
   from `UNSUPPORTED_CURRENT_LAYERS` only once the extra hop is handled.
4. If the dependency block has a different branch layout than `mixed4d` (see
   `constants/branches.py`), add a new ordered branch list rather than hardcoding channel
   boundaries inline.
