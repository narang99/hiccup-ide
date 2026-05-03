from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404
from typing import Optional, List as ListType
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
    WorkGraphMeta,
)
from .helpers import get_min_max, apply_algorithm

router = Router()


def get_work_graph_context(
    input_obj: Input, work_alias: Optional[str], graph_alias: Optional[str]
):
    if not (work_alias and graph_alias):
        return None, None

    work_graph = WorkGraph.objects.filter(
        work__input=input_obj, work__name=work_alias, alias=graph_alias
    ).first()

    if not work_graph:
        return None, None

    return work_graph, WorkGraphMeta(work_alias=work_alias, graph_alias=graph_alias)


def get_saliency_maps_by_coordinates(
    input_obj: Input,
    coordinates: ListType[str],
    work_alias: Optional[str] = None,
    graph_alias: Optional[str] = None,
    allow_missing: bool = False,
) -> ListType[SaliencyMapOut]:
    # 1. Resolve work context
    work_graph, work_meta = get_work_graph_context(input_obj, work_alias, graph_alias)

    results_dict = {}

    # 2. Query WorkSaliencyMap if context exists
    if work_graph:
        work_maps = WorkSaliencyMap.objects.filter(
            graph=work_graph, coordinate__in=coordinates
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
                work_graph=work_meta,
            )

    # 3. Find missing coordinates and query base SaliencyMap
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
                work_graph=None,
            )

    # 4. Verify all coordinates are found
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
    graph_alias: Optional[str] = None,
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
        input_obj, coordinates, work_alias, graph_alias
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
    graph_alias: Optional[str] = None,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    items = get_saliency_maps_by_coordinates(
        input_obj, data.coordinates, work_alias, graph_alias
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
    graph_alias: Optional[str] = None,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    items = get_saliency_maps_by_coordinates(
        input_obj, [coordinate], work_alias, graph_alias
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
    graph_alias: Optional[str] = None,
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all saliency maps for the requested coordinates using the helper
    items = get_saliency_maps_by_coordinates(
        input_obj, data.coordinates, work_alias, graph_alias, allow_missing=True
    )

    saliency_maps = [item.data for item in items]
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
    # TOTAL_LAYERS = ["layers.3", "layers.2", "layers.1", "layers.0", "inputs"]
    TOTAL_LAYERS = ["layers.3", "layers.2", "layers.1", "layers.0"]

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
