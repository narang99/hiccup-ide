import functools
from django.shortcuts import get_object_or_404
from django.db import transaction
from neural_data.models import (
    Model,
    Input,
    Work,
    WorkGraph,
    TempPruneSaliencyMap,
)
from neural_data.schemas import (
    BatchWorkSaliencyMapIn,
    CoordinateAlgorithmIn,
)
from neural_data.api.prune.helpers import apply_algorithm


def apply_pruning_to_current_layer(
    input_obj: Input, graph: WorkGraph, items: list[CoordinateAlgorithmIn]
):
    updated_count = 0
    current_layer_name = None

    for item in items:
        temp_map = get_object_or_404(
            TempPruneSaliencyMap,
            input=input_obj,
            coordinate=item.coordinate,
            graph=graph,
        )

        if current_layer_name is None:
            current_layer_name = temp_map.layer_name
        elif current_layer_name != temp_map.layer_name:
            raise ValueError(
                f"All coordinates must belong to the same layer. Expected {current_layer_name}, got {temp_map.layer_name}"
            )

        # # Apply algorithm to the ORIGINAL base data as per requirements
        # orig_map = get_object_or_404(
        #     TempPruneSaliencyMap, input=input_obj, coordinate=item.coordinate
        # )
        filtered_data = apply_algorithm(temp_map.data, item.algorithm)

        temp_map.data = filtered_data
        temp_map.is_modified = True
        temp_map.save()
        updated_count += 1

    return updated_count, current_layer_name


def reconstruct_layer_tensor(graph: WorkGraph, layer_name: str):
    # We need all output_channel coordinates for this layer to form the [1, C, H, W] tensor
    import torch

    layer_coords = TempPruneSaliencyMap.objects.filter(
        graph=graph, layer_name=layer_name, coordinate_type="output_channel"
    )

    coord_list = list(layer_coords)
    if not coord_list:
        return None

    sample_shape = coord_list[0].shape  # [H, W]
    # Determine max channel to size the tensor correctly
    max_ch = max(
        tm.output_channel for tm in coord_list if tm.output_channel is not None
    )
    num_channels = max_ch + 1

    tensor = torch.zeros((1, num_channels, sample_shape[0], sample_shape[1]))
    for tm in coord_list:
        if tm.output_channel is not None:
            tensor[0, tm.output_channel] = torch.tensor(tm.data)
    return tensor


def update_upstream_temp_maps(
    graph: WorkGraph, current_layer_name: str, new_coords_dict: dict
):
    TOTAL_LAYERS_ORDER = ["layers.3", "layers.2", "layers.1", "layers.0", "x"]
    try:
        current_idx = TOTAL_LAYERS_ORDER.index(current_layer_name)
        upstream_layer_names = set(TOTAL_LAYERS_ORDER[current_idx + 1 :])
    except ValueError:
        upstream_layer_names = set()

    # Bulk update TempPruneSaliencyMap for all upstream coordinates found in new_coords_dict
    temp_maps_to_update = []
    relevant_temp_maps = TempPruneSaliencyMap.objects.filter(
        graph=graph, coordinate__in=new_coords_dict.keys()
    )

    for tm in relevant_temp_maps:
        if tm.layer_name in upstream_layer_names or tm.coordinate == "x.out_0":
            coord_update_data = new_coords_dict[tm.coordinate]
            tm.data = coord_update_data["data"]
            tm.shape = coord_update_data["shape"]
            tm.output_channel = coord_update_data.get("output_channel")
            tm.input_channel = coord_update_data.get("input_channel")
            # We don't mark upstream as 'is_modified' because they haven't been user-pruned yet
            temp_maps_to_update.append(tm)

    if temp_maps_to_update:
        TempPruneSaliencyMap.objects.bulk_update(
            temp_maps_to_update, ["data", "shape", "output_channel", "input_channel"]
        )


@functools.lru_cache(maxsize=1)
def _get_cached_mnist_model(model_obj: Model):
    from pt_to_api.mnist import SimpleMNIST

    pt_file = model_obj.load_pt_file()
    if pt_file is None:
        raise FileNotFoundError(f"Model pt_file not found for model {model_obj.alias}")

    model = SimpleMNIST()
    # If the pt_file is a state dict, load it. If it's a full model, we might need different logic.
    # For now, assuming it's a state dict as per existing code.
    model.load_state_dict(pt_file)
    model = model.to("cpu")
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def _get_cached_mnist_input_tensor(input_obj: Input):
    import torch

    pt_file = input_obj.load_pt_file().to("cpu")
    if pt_file is None:
        raise FileNotFoundError(f"Input pt_file not found for input {input_obj.alias}")
    return pt_file


def recalculate_upstream_saliency(
    input_obj: Input, graph: WorkGraph, current_layer_name: str
):
    from pt_to_api.contrib_processor import process_contribs_to_coordinates
    from pt_to_api.mnist import get_contribs_for_inp_vectorized

    try:
        model = _get_cached_mnist_model(input_obj.model)
        batch_inp_tens = _get_cached_mnist_input_tensor(input_obj)

        # Reconstruct the current layer's pruned contributions into a tensor
        last_layer_contribs = reconstruct_layer_tensor(graph, current_layer_name)
        if last_layer_contribs is None:
            raise ValueError(
                f"Could not reconstruct tensor for layer {current_layer_name} - no output channel coordinates found"
            )

        # Run re-propagation
        total_contribs, _, _ = get_contribs_for_inp_vectorized(
            batch_inp_tens, model, last_layer_contribs, current_layer_name, "cpu"
        )

        # Convert new contributions to coordinates
        new_coords_dict = process_contribs_to_coordinates(total_contribs, sample_idx=0)

        # Update all upstream layers in the scratchpad
        update_upstream_temp_maps(graph, current_layer_name, new_coords_dict)

    except Exception as e:
        # Re-raise the exception with context about what operation failed
        raise RuntimeError(
            f"Re-propagation failed for layer {current_layer_name}: {str(e)}"
        ) from e


def inner_create_batch_work_saliency_maps(
    model_alias: str,
    input_alias: str,
    workflow_name: str,
    payload: BatchWorkSaliencyMapIn,
):
    # this is not very efficient obviously
    # the problem is that for now we are committing the current layer contribs to backend
    # then we are using the backend to reconstruct the layers
    # in case of any failure after committing, tis problem
    # im not sure about changing the order of commit right now, its fine
    with transaction.atomic():
        input_obj = get_object_or_404(
            Input, model__alias=model_alias, alias=input_alias
        )
        work = get_object_or_404(Work, input=input_obj, name=workflow_name)
        graph = get_object_or_404(WorkGraph, work=work)

        if not payload.items:
            return {"created": 0, "updated": 0}

        # 1. Update the current layer in TempPruneSaliencyMap with pruned data
        updated_count, current_layer_name = apply_pruning_to_current_layer(
            input_obj, graph, payload.items
        )

        if current_layer_name is None:
            raise ValueError("Failed to determine current layer name from coordinates")

        # 2. Re-calculate upstream contributions if we are not at the input layer
        if current_layer_name != "x":
            recalculate_upstream_saliency(input_obj, graph, current_layer_name)

        return {"created": 0, "updated": updated_count}
