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

## Why the pad/pool bookkeeping exists, and why it's inert for `CifarInception`

The metadata tables all feed **one flow**: `NeuronParentAnalyser` + its report
generation (`report_data`/`report_render`/`report_assets`/`feature_viz`). That
flow answers "when neuron `(current_layer, current_channel)` fires at `(y,x)`,
which upstream neurons explain it, and which clusters (from the `s0`→`s4`
pipeline) do they belong to?". It works off a single tensor — the concatenated
input to a 1x1 conv — and the tables recover what that tensor alone doesn't say:

- `BRANCHES_BY_CURRENT_LAYER` → `FlattenedChannelMap` (`iter_dependency_coords`):
  demuxes a flattened channel index in the concat back to `(branch_layer, channel)`.
- `F_PAD_MANUAL_BY_CURRENT_LAYER` (`dep_coords`): translates a spatial coord from
  the current conv's captured-input frame to the dep's own output frame.
- `LAYER_NAME_BY_SHAPE` (`report_assets`): grid shape to reshape a flattened
  weight*patch vector for the report image.
- `_DEP_LAYER_HOOK_SOURCE` (`feature_viz`): which module to hook (+ pad/pool to
  replay) to feature-viz-reconstruct a dep neuron for the report thumbnail.

**Key point for `CifarInception` (`src/olt/models/cifar_inception.py`):** the
whole `F_PAD_MANUAL` / `dep_coords` offset and the `_DEP_LAYER_HOOK_SOURCE`
`prep_fn` (pad/pool replay) machinery only existed because lucent's InceptionV1
does its pads/pools as **functional ops** (`F.pad`/`F.max_pool2d`) that aren't
hookable modules — so the analyser had to manually replay them to line up
coordinate frames. `CifarInception` folds every pad into the conv's own
`padding=` and uses `nn.MaxPool2d`, so:

- `F_PAD_MANUAL_BY_CURRENT_LAYER` entries are all `(0,0)` — the captured conv
  input already equals the dep-output frame, and `receptive_block` shifts by
  `layer.padding` on its own. `dep_coords` is an identity no-op.
- `_DEP_LAYER_HOOK_SOURCE` entries all have `prep_fn=None` — a conv reads a module
  output directly, nothing to replay (`padding` comes from `get_layer_params`).

So when wiring the analysis pass for `CifarInception`, do NOT re-derive pad
tables: pass `f_pad=(0,0)` and `prep_fn=None`. The only bookkeeping that's real
for this arch (inherent to the channel-concat, not to hooks) is `BLOCK_BRANCHES`
(demux) and `LAYER_NAME_BY_SHAPE` (report viz). The `pool_reduce`-style
limitation still holds — a maxpool between the block input and a `pool_reduce`
conv has its own receptive field `dep_coords` can't translate — but that was
never about padding, and `pool_reduce` isn't in `ANALYSABLE_1X1_LAYERS`.
