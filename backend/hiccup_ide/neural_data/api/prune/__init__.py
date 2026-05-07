from ninja import Router
from django.shortcuts import get_object_or_404
from neural_data.models import (
    Input,
    SaliencyMap,
    Work,
    WorkGraph,
    WorkSaliencyMap,
    TempPruneSaliencyMap,
)
from neural_data.schemas import (
    BatchWorkSaliencyMapIn,
    PruningStatusOut,
)
from .update_batch import inner_create_batch_work_saliency_maps

router = Router()


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
    temp_maps = TempPruneSaliencyMap.objects.filter(graph=graph)

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
        temp_qs.filter(is_modified=True).values_list("layer_name", flat=True).distinct()
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
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/saliency_maps/"
)
def create_batch_work_saliency_maps(
    request,
    model_alias: str,
    input_alias: str,
    workflow_name: str,
    payload: BatchWorkSaliencyMapIn,
):
    return inner_create_batch_work_saliency_maps(
        model_alias, input_alias, workflow_name, payload
    )