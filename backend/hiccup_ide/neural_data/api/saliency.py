from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404
from ..models import (
    Input,
    SaliencyMap,
    Work,
    WorkGraph,
    WorkSaliencyMap,
)
from ..schemas import (
    SaliencyMapOut,
    SaliencyMapListOut,
    BatchSaliencyMapsIn,
    NodeStatsOut,
    BatchWorkSaliencyMapIn,
)
from .helpers import get_min_max, apply_algorithm

router = Router()

@router.get(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/layers/{layer_name}/",
    response=SaliencyMapListOut,
)
def get_layer_saliency_maps(
    request, model_alias: str, input_alias: str, layer_name: str
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all saliency maps for coordinates that start with the layer name
    saliency_maps = SaliencyMap.objects.filter(
        input=input_obj, coordinate__startswith=f"{layer_name}."
    ).order_by("coordinate")

    if not saliency_maps.exists():
        raise Http404(f"No saliency maps found for layer '{layer_name}'")

    return {"items": list(saliency_maps)}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/batch/",
    response=SaliencyMapListOut,
)
def get_batch_saliency_maps(
    request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all saliency maps for the requested coordinates
    saliency_maps = SaliencyMap.objects.filter(
        input=input_obj, coordinate__in=data.coordinates
    ).order_by("coordinate")

    return {"items": list(saliency_maps)}


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/single/{coordinate}/",
    response=SaliencyMapOut,
)
def get_saliency_map(request, model_alias: str, input_alias: str, coordinate: str):
    saliency_map = get_object_or_404(
        SaliencyMap,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate,
    )
    return saliency_map


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/stats/",
    response=NodeStatsOut,
)
def get_saliency_maps_stats(
    request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all saliency maps for the requested coordinates
    saliency_maps = list(
        SaliencyMap.objects.filter(
            input=input_obj, coordinate__in=data.coordinates
        ).values_list("data", flat=True)
    )

    min_val, max_val = get_min_max(saliency_maps)
    return {"min": min_val, "max": max_val}


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/graphs/{graph_alias}/status/"
)
def get_workflow_pruning_status(
    request, model_alias: str, input_alias: str, workflow_name: str, graph_alias: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get the work/workflow
    work = get_object_or_404(Work, input=input_obj, name=workflow_name)

    # Get the work graph
    work_graph = get_object_or_404(WorkGraph, work=work, alias=graph_alias)

    # Hardcoded total layers in specific order as requested
    TOTAL_LAYERS = ["layers.3", "layers.2", "layers.1", "layers.0", "inputs"]

    # Get all unique layer_names present in WorkSaliencyMap for this graph
    done_layers_set = set(
        WorkSaliencyMap.objects.filter(graph=work_graph, layer_name__isnull=False)
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

    return {"layers": {"done": done_layers, "total": TOTAL_LAYERS}}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/graphs/{graph_alias}/"
)
def create_or_update_work_graph(
    request, model_alias: str, input_alias: str, workflow_name: str, graph_alias: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)

    # Create or update the work graph
    work_graph, created = WorkGraph.objects.get_or_create(work=work, alias=graph_alias)

    return {"id": work_graph.pk, "alias": work_graph.alias, "created": created}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
)
def create_batch_work_saliency_maps(
    request,
    model_alias: str,
    input_alias: str,
    workflow_name: str,
    graph_alias: str,
    payload: BatchWorkSaliencyMapIn,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=workflow_name)
    graph = get_object_or_404(WorkGraph, work=work, alias=graph_alias)

    created_count = 0
    updated_count = 0

    for item in payload.items:
        # Get original saliency map
        orig_map = get_object_or_404(
            SaliencyMap, input=input_obj, coordinate=item.coordinate
        )

        # Apply algorithm
        filtered_data = apply_algorithm(orig_map.data, item.algorithm)

        # Create or update WorkSaliencyMap
        work_map, created = WorkSaliencyMap.objects.update_or_create(
            input=input_obj,
            coordinate=item.coordinate,
            graph=graph,
            defaults={
                "data": filtered_data,
                "shape": orig_map.shape,
                "layer_name": orig_map.layer_name,
                "coordinate_type": orig_map.coordinate_type,
                "data_type": orig_map.data_type,
            },
        )

        if created:
            created_count += 1
        else:
            updated_count += 1

    return {"created": created_count, "updated": updated_count}
