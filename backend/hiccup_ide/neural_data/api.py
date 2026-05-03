from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404

from .models import (
    Model,
    Input,
    Activation,
    SaliencyMap,
    Weight,
    Work,
    LayerThreshold,
    WorkGraph,
    WorkSaliencyMap,
)
from .schemas import (
    ModelOut,
    InputOut,
    ActivationOut,
    SaliencyMapOut,
    WeightOut,
    LayerSaliencyMapsOut,
    BatchSaliencyMapsIn,
    NodeStatsOut,
    LayerThresholdIn,
    LayerThresholdOut,
)

router = Router()


@router.get("/")
def root(request):
    return {"message": "Hiccup IDE API"}


@router.get("/models/{model_alias}/", response=ModelOut)
def get_model(request, model_alias: str):
    model = get_object_or_404(Model, alias=model_alias)
    return model


@router.get("/models/{model_alias}/inputs/{input_alias}/", response=InputOut)
def get_input(request, model_alias: str, input_alias: str):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    return input_obj


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/activations/single/{coordinate}/",
    response=ActivationOut,
)
def get_activation(request, model_alias: str, input_alias: str, coordinate: str):
    activation = get_object_or_404(
        Activation,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate,
    )
    return activation


@router.get("/models/{model_alias}/weights/single/{coordinate}/", response=WeightOut)
def get_weight(request, model_alias: str, coordinate: str):
    weight = get_object_or_404(Weight, model__alias=model_alias, coordinate=coordinate)
    return weight


@router.get(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/layers/{layer_name}/",
    response=LayerSaliencyMapsOut,
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

    return {"layer_name": layer_name, "saliency_maps": list(saliency_maps)}


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/saliency_maps/batch/",
    response=LayerSaliencyMapsOut,
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

    return {"layer_name": "batch", "saliency_maps": list(saliency_maps)}


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


def _get_min_max_in_list_of_lists(data_lists):
    main_min, main_max = None, None
    for data in data_lists:
        if not data:
            continue
        cur_min, cur_max = max(data), min(data)
        print("currrrrrr", cur_min)
        print("dataaaaaaaa", data)
        if main_min is None or cur_min < main_min:
            main_min = cur_min
        if main_max is None or cur_max > main_max:
            main_max = cur_max
    print("mainnnnnn", main_min)
    return main_min, main_max


def get_min_max(data_list):
    """Helper to find min and max in a list of nested data (JSON)."""
    # keep going down until we find a list of elements
    # then simply run min/max on them and return them

    def _find_min_max(items):
        if isinstance(items, list):
            if not items:
                return 0.0, 0.0
            elif isinstance(items[0], (int, float)):
                return min(items), max(items)
            else:
                main_min, main_max = None, None
                for item in items:
                    cur_min, cur_max = _find_min_max(item)
                    if main_min is None or cur_min < main_min:
                        main_min = cur_min
                    if main_max is None or cur_max > main_max:
                        main_max = cur_max
                return main_min, main_max
        elif isinstance(items, (int, float)):
            return items, items
        else:
            raise Exception(
                f"only lists or floats are allowed for finding min-max, got={type(items)} items={items}"
            )

    return _find_min_max(data_list)


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/activations/stats/",
    response=NodeStatsOut,
)
def get_activations_stats(
    request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn
):
    # Verify input exists
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get all activations for the requested coordinates
    activations = list(
        Activation.objects.filter(
            input=input_obj, coordinate__in=data.coordinates
        ).values_list("data", flat=True)
    )

    min_val, max_val = get_min_max(activations)
    return {"min": min_val, "max": max_val}


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
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/thresholds/",
    response=list[LayerThresholdOut],
)
def get_workflow_thresholds(
    request, model_alias: str, input_alias: str, workflow_name: str
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)

    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)

    # Get all thresholds for this work and model
    thresholds = LayerThreshold.objects.filter(work=work, model__alias=model_alias)

    return list(thresholds)


@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/thresholds/",
    response=LayerThresholdOut,
)
def save_workflow_threshold(
    request,
    model_alias: str,
    input_alias: str,
    workflow_name: str,
    data: LayerThresholdIn,
):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    model_obj = get_object_or_404(Model, alias=model_alias)

    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)

    # Create or update the threshold
    threshold, created = LayerThreshold.objects.update_or_create(
        work=work,
        model=model_obj,
        layer_id=data.layer_id,
        defaults={"slider_value": data.slider_value, "algorithm": data.algorithm},
    )

    return threshold


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

@router.post("/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/graphs/{graph_alias}/")
def create_or_update_work_graph(request, model_alias: str, input_alias: str, workflow_name: str, graph_alias: str):
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    
    # Create or update the work graph
    work_graph, created = WorkGraph.objects.get_or_create(
        work=work,
        alias=graph_alias
    )
    
    return {
        "id": work_graph.pk,
        "alias": work_graph.alias,
        "created": created
    }