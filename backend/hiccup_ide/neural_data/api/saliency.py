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
    TempPruneSaliencyMap,
)
from ..schemas import (
    SaliencyMapOut,
    SaliencyMapListOut,
    BatchSaliencyMapsIn,
    NodeStatsOut,
    WorkGraphMeta,
    MarkSliceDoneIn,
    UpdateSliceStateIn,
    SliceStatusOut,
)
from .helpers import get_min_max

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
    "/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/mark-done/"
)
def mark_slice_done(
    request, 
    model_alias: str, 
    input_alias: str, 
    work_alias: str, 
    data: MarkSliceDoneIn
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=work_alias)
    work_graph = get_object_or_404(WorkGraph, work=work)
    saliency_map = get_object_or_404(WorkSaliencyMap, graph=work_graph, coordinate=data.coordinate)
    saliency_map.is_done = True
    saliency_map.save()
    return {"success": True, "coordinate": data.coordinate, "is_done": True}
    

@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/unmark-done/"
)
def unmark_slice_done(
    request, 
    model_alias: str, 
    input_alias: str, 
    work_alias: str, 
    data: MarkSliceDoneIn
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=work_alias)
    work_graph = get_object_or_404(WorkGraph, work=work)
    saliency_map = get_object_or_404(WorkSaliencyMap, graph=work_graph, coordinate=data.coordinate)
    saliency_map.is_done = False
    saliency_map.save()
    return {"success": True, "coordinate": data.coordinate, "is_done": False}
    

@router.get(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/slice-status/{coordinate}/",
    response=SliceStatusOut
)
def get_slice_status(
    request,
    model_alias: str,
    input_alias: str,
    work_alias: str,
    coordinate: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=work_alias)
    work_graph = get_object_or_404(WorkGraph, work=work)
    saliency_map = get_object_or_404(WorkSaliencyMap, graph=work_graph, coordinate=coordinate)
    
    # Return both old is_done field and new state field
    return {
        "coordinate": coordinate, 
        "is_done": saliency_map.is_done,
        "state": saliency_map.slice_state
    }


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/update-slice-state/",
    response=SliceStatusOut
)
def update_slice_state(
    request,
    model_alias: str,
    input_alias: str,
    work_alias: str,
    data: UpdateSliceStateIn
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=work_alias)
    work_graph = get_object_or_404(WorkGraph, work=work)
    saliency_map = get_object_or_404(WorkSaliencyMap, graph=work_graph, coordinate=data.coordinate)
    
    # Update the slice state
    saliency_map.slice_state = data.state
    
    # Update is_done field based on the state (done only when state is 'done')
    saliency_map.is_done = (data.state == 'done')
    
    saliency_map.save()
    
    return {
        "coordinate": data.coordinate,
        "is_done": saliency_map.is_done,
        "state": saliency_map.slice_state
    }

