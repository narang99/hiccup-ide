import functools
from ninja import Router
import os
from django.shortcuts import get_object_or_404
from django.http import Http404
from django.db import transaction
from typing import Optional, List as ListType
from ..models import (
    Model,
    Input,
    SaliencyMap,
    Work,
    WorkGraph,
    WorkSaliencyMap,
    TempPruneSaliencyMap,
)
from ..schemas import (
    SaliencyMapOut,
    SaliencyMapListOut,
    BatchSaliencyMapsIn,
    NodeStatsOut,
    BatchWorkSaliencyMapIn,
    WorkGraphMeta,
    PruningStatusOut,
    CoordinateAlgorithmIn,
)
from .helpers import get_min_max, apply_algorithm

router = Router()


def get_work_graph_context(input_obj: Input, work_alias: Optional[str]):
    if not work_alias:
        return None, None

    try:
        work = Work.objects.get(input=input_obj, name=work_alias)
        work_graph = WorkGraph.objects.get(work=work)
        return work_graph, WorkGraphMeta(work_alias=work_alias)
    except (Work.DoesNotExist, WorkGraph.DoesNotExist):
        return None, None


def get_saliency_maps_by_coordinates(
    input_obj: Input,
    coordinates: ListType[str],
    work_alias: Optional[str] = None,
    pruned: bool = False,
    allow_missing: bool = False,
) -> ListType[SaliencyMapOut]:
    # 1. Resolve work context
    work_graph, work_meta = get_work_graph_context(input_obj, work_alias)

    results_dict = {}

    # When pruned=True and work context exists, prioritize pruned data
    if pruned and work_graph:
        # 2. Query TempPruneSaliencyMap first (Scratchpad priority)
        temp_maps = TempPruneSaliencyMap.objects.filter(
            graph=work_graph, coordinate__in=coordinates
        )
        for tm in temp_maps:
            results_dict[tm.coordinate] = SaliencyMapOut(
                id=tm.pk,
                coordinate=tm.coordinate,
                layer_name=tm.layer_name,
                data=tm.data,
                shape=tm.shape,
                coordinate_type=tm.coordinate_type,
                data_type=tm.data_type,
                output_channel=tm.output_channel,
                input_channel=tm.input_channel,
                work_graph=work_meta,
            )

        # 3. Query WorkSaliencyMap for coordinates not found in Temp
        remaining_coords = [c for c in coordinates if c not in results_dict]
        if remaining_coords:
            work_maps = WorkSaliencyMap.objects.filter(
                graph=work_graph, coordinate__in=remaining_coords
            )
            for wm in work_maps:
                results_dict[wm.coordinate] = SaliencyMapOut(
                    id=wm.pk,
                    coordinate=wm.coordinate,
                    layer_name=wm.layer_name,
                    data=wm.data,
                    shape=wm.shape,
                    coordinate_type=wm.coordinate_type,
                    data_type=wm.data_type,
                    output_channel=wm.output_channel,
                    input_channel=wm.input_channel,
                    work_graph=work_meta,
                )

    # 4. For unpruned data or when no work context, always query base SaliencyMap
    remaining_coords = [c for c in coordinates if c not in results_dict]
    if remaining_coords:
        base_maps = SaliencyMap.objects.filter(
            input=input_obj, coordinate__in=remaining_coords
        )
        for bm in base_maps:
            results_dict[bm.coordinate] = SaliencyMapOut(
                id=bm.pk,
                coordinate=bm.coordinate,
                layer_name=bm.layer_name,
                data=bm.data,
                shape=bm.shape,
                coordinate_type=bm.coordinate_type,
                data_type=bm.data_type,
                output_channel=bm.output_channel,
                input_channel=bm.input_channel,
                work_graph=None,
            )

    # 5. Verify all coordinates are found
    if not allow_missing and len(results_dict) != len(set(coordinates)):
        missing = set(coordinates) - set(results_dict.keys())
        raise Http404(f"Saliency maps not found for coordinates: {list(missing)}")

    # Return in the original order requested (filter out missing if allowed)
    return [results_dict[c] for c in coordinates if c in results_dict]


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/layers/{layer_name}/",
    response=SaliencyMapListOut,
)
def get_layer_saliency_maps(
    request,
    model_alias: str,
    input_alias: str,
    layer_name: str,
    work_alias: Optional[str] = None,
    pruned: bool = False,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # First, get all coordinate names for this layer from base saliency maps
    coordinates = list(
        SaliencyMap.objects.filter(
            input=input_obj, coordinate__startswith=f"{layer_name}."
        )
        .order_by("coordinate")
        .values_list("coordinate", flat=True)
    )

    if not coordinates:
        raise Http404(f"No saliency maps found for layer '{layer_name}'")

    items = get_saliency_maps_by_coordinates(
        input_obj, coordinates, work_alias, pruned
    )
    return {"items": items}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/batch/",
    response=SaliencyMapListOut,
)
def get_batch_saliency_maps(
    request,
    model_alias: str,
    input_alias: str,
    data: BatchSaliencyMapsIn,
    work_alias: Optional[str] = None,
    pruned: bool = False,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    items = get_saliency_maps_by_coordinates(
        input_obj, data.coordinates, work_alias, pruned
    )
    return {"items": items}


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/single/{coordinate}/",
    response=SaliencyMapOut,
)
def get_saliency_map(
    request,
    model_alias: str,
    input_alias: str,
    coordinate: str,
    work_alias: Optional[str] = None,
    pruned: bool = False,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    items = get_saliency_maps_by_coordinates(
        input_obj, [coordinate], work_alias, pruned
    )
    return items[0]


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/stats/",
    response=NodeStatsOut,
)
def get_saliency_maps_stats(
    request,
    model_alias: str,
    input_alias: str,
    data: BatchSaliencyMapsIn,
    work_alias: Optional[str] = None,
    pruned: bool = False,
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all saliency maps for the requested coordinates using the helper
    items = get_saliency_maps_by_coordinates(
        input_obj, data.coordinates, work_alias, pruned, allow_missing=True
    )

    saliency_maps = [item.data for item in items]
    min_val, max_val = get_min_max(saliency_maps)
    return {"min": min_val, "max": max_val}


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/status/",
    response=PruningStatusOut,
)
def get_workflow_pruning_status(
    request, model_alias: str, input_alias: str, workflow_name: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get the work/workflow
    work = get_object_or_404(Work, input=input_obj, name=workflow_name)

    # Get the work graph (OneToOne relationship)
    work_graph = get_object_or_404(WorkGraph, work=work)

    # Hardcoded total layers in specific order as requested
    TOTAL_LAYERS = ["layers.3", "layers.2", "layers.1", "layers.0", "x"]

    # Prune status only checks TempPruneSaliencyMap (active session)
    temp_qs = TempPruneSaliencyMap.objects.filter(graph=work_graph)
    session_active = temp_qs.exists()

    # Only consider temp maps that have been modified by the user
    done_layers_set = set(
        temp_qs.filter(is_modified=True)
        .values_list("layer_name", flat=True)
        .distinct()
    )

    # Build done list maintaining strict order
    done_layers = []
    for layer in TOTAL_LAYERS:
        if layer in done_layers_set:
            done_layers.append(layer)
        else:
            # Once we find a gap, we stop adding to done_layers to ensure sequential order
            break

    return {
        "layers": {"done": done_layers, "total": TOTAL_LAYERS},
        "session_active": session_active,
    }


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/"
)
def create_or_update_work_graph(
    request, model_alias: str, input_alias: str, workflow_name: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)

    # Create or update the work graph (OneToOne relationship)
    work_graph, created = WorkGraph.objects.get_or_create(work=work)

    return {"id": work_graph.pk, "created": created}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/start_pruning/"
)
def initialize_pruning_session(
    request, model_alias: str, input_alias: str, workflow_name: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    graph, _ = WorkGraph.objects.get_or_create(work=work)

    # Clear existing temp maps for this graph
    TempPruneSaliencyMap.objects.filter(graph=graph).delete()

    # Clone all base SaliencyMaps to TempPruneSaliencyMap
    base_maps = SaliencyMap.objects.filter(input=input_obj)
    
    temp_maps = [
        TempPruneSaliencyMap(
            input=input_obj,
            coordinate=bm.coordinate,
            graph=graph,
            data=bm.data,
            shape=bm.shape,
            layer_name=bm.layer_name,
            coordinate_type=bm.coordinate_type,
            data_type=bm.data_type,
            output_channel=bm.output_channel,
            input_channel=bm.input_channel,
            is_modified=False,
        )
        for bm in base_maps
    ]
    
    TempPruneSaliencyMap.objects.bulk_create(temp_maps)

    return {"status": "started", "cloned_count": len(temp_maps)}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/finalize_pruning/"
)
def finalize_pruning_session(
    request, model_alias: str, input_alias: str, workflow_name: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=workflow_name)
    graph = get_object_or_404(WorkGraph, work=work)

    # Only commit modified temp maps to WorkSaliencyMap
    temp_maps = TempPruneSaliencyMap.objects.filter(graph=graph, is_modified=True)
    
    if not temp_maps.exists():
         # Clean up anyway if no modifications were made
         TempPruneSaliencyMap.objects.filter(graph=graph).delete()
         return {"status": "finalized", "committed_count": 0}

    committed_count = 0
    for tm in temp_maps:
        WorkSaliencyMap.objects.update_or_create(
            input=input_obj,
            coordinate=tm.coordinate,
            graph=graph,
            defaults={
                "data": tm.data,
                "shape": tm.shape,
                "layer_name": tm.layer_name,
                "coordinate_type": tm.coordinate_type,
                "data_type": tm.data_type,
                "output_channel": tm.output_channel,
                "input_channel": tm.input_channel,
            },
        )
        committed_count += 1

    # Clear temp maps
    TempPruneSaliencyMap.objects.filter(graph=graph).delete()

    return {"status": "finalized", "committed_count": committed_count}


def apply_pruning_to_current_layer(
    input_obj: Input, graph: WorkGraph, items: list[CoordinateAlgorithmIn]
):
    updated_count = 0
    current_layer_name = None

    for item in items:
        temp_map = get_object_or_404(
            TempPruneSaliencyMap, input=input_obj, coordinate=item.coordinate, graph=graph
        )

        if current_layer_name is None:
            current_layer_name = temp_map.layer_name
        elif current_layer_name != temp_map.layer_name:
            raise ValueError(f"All coordinates must belong to the same layer. Expected {current_layer_name}, got {temp_map.layer_name}")

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
            raise ValueError(f"Could not reconstruct tensor for layer {current_layer_name} - no output channel coordinates found")

        # Run re-propagation
        total_contribs, _, _ = get_contribs_for_inp_vectorized(
            batch_inp_tens, model, last_layer_contribs, current_layer_name, "cpu"
        )

        # Convert new contributions to coordinates
        new_coords_dict = process_contribs_to_coordinates(
            total_contribs, sample_idx=0
        )

        # Update all upstream layers in the scratchpad
        update_upstream_temp_maps(graph, current_layer_name, new_coords_dict)

    except Exception as e:
        # Re-raise the exception with context about what operation failed
        raise RuntimeError(f"Re-propagation failed for layer {current_layer_name}: {str(e)}") from e


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/saliency_maps/"
)
def create_batch_work_saliency_maps(
    request,
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
        input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
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
