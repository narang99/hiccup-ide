from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404

from .models import Model, Input, Activation, SaliencyMap, Weight, Work, LayerThreshold, WorkGraph, WorkSaliencyMap
from .schemas import (
    ModelOut, InputOut, ActivationOut, SaliencyMapOut, 
    WeightOut, LayerSaliencyMapsOut, BatchSaliencyMapsIn,
    NodeStatsOut, LayerThresholdIn, LayerThresholdOut
)

router = Router()

def get_saliency_maps_with_pruned_graph_support(input_obj, coordinates, pruned_graph_alias=None):
    """
    Get saliency maps with pruned graph support. Falls back to original saliency maps
    if pruned graph data is not present.
    """
    if pruned_graph_alias:
        # Try to get work graph
        try:
            work_graph = WorkGraph.objects.get(
                work__input=input_obj,
                alias=pruned_graph_alias
            )
            
            # Get WorkSaliencyMap entries for the coordinates
            work_saliency_maps = WorkSaliencyMap.objects.filter(
                graph=work_graph,
                coordinate__in=coordinates
            )
            
            # Convert to dict for easy lookup
            work_saliency_dict = {wsm.coordinate: wsm for wsm in work_saliency_maps}
            
            # Build result list - use work saliency map if available, else fallback to original
            result_saliency_maps = []
            for coordinate in coordinates:
                if coordinate in work_saliency_dict:
                    result_saliency_maps.append(work_saliency_dict[coordinate])
                else:
                    # Fallback to original saliency map
                    try:
                        original_saliency = SaliencyMap.objects.get(
                            input=input_obj,
                            coordinate=coordinate
                        )
                        result_saliency_maps.append(original_saliency)
                    except SaliencyMap.DoesNotExist:
                        pass  # Skip if neither exists
            
            return result_saliency_maps
            
        except WorkGraph.DoesNotExist:
            pass  # Fall through to original saliency maps
    
    # Default behavior - get original saliency maps
    return SaliencyMap.objects.filter(
        input=input_obj,
        coordinate__in=coordinates
    )

@router.get("/")
def root(request):
    return {"message": "Hiccup IDE API"}

@router.get("/models/{model_alias}/", response=ModelOut)
def get_model(request, model_alias: str):
    model = get_object_or_404(Model, alias=model_alias)
    return model

@router.get("/models/{model_alias}/inputs/{input_alias}/", response=InputOut)
def get_input(request, model_alias: str, input_alias: str):
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    return input_obj

@router.get("/models/{model_alias}/inputs/{input_alias}/activations/single/{coordinate}/", response=ActivationOut)
def get_activation(request, model_alias: str, input_alias: str, coordinate: str):
    activation = get_object_or_404(
        Activation,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate
    )
    return activation

@router.get("/models/{model_alias}/weights/single/{coordinate}/", response=WeightOut)
def get_weight(request, model_alias: str, coordinate: str):
    weight = get_object_or_404(
        Weight,
        model__alias=model_alias,
        coordinate=coordinate
    )
    return weight

@router.get("/models/{model_alias}/inputs/{input_alias}/saliency_maps/layers/{layer_name}/", response=LayerSaliencyMapsOut)
def get_layer_saliency_maps(request, model_alias: str, input_alias: str, layer_name: str, pruned_graph: str = None):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    if pruned_graph:
        # Use pruned graph logic with fallback
        try:
            work_graph = WorkGraph.objects.get(
                work__input=input_obj,
                alias=pruned_graph
            )
            
            # Get WorkSaliencyMap entries for coordinates that start with the layer name
            work_saliency_maps = WorkSaliencyMap.objects.filter(
                graph=work_graph,
                coordinate__startswith=f"{layer_name}."
            ).order_by('coordinate')
            
            # Get original saliency maps for fallback
            original_saliency_maps = SaliencyMap.objects.filter(
                input=input_obj,
                coordinate__startswith=f"{layer_name}."
            ).order_by('coordinate')
            
            # Create a dict of work saliency maps by coordinate
            work_saliency_dict = {wsm.coordinate: wsm for wsm in work_saliency_maps}
            
            # Build result with fallback logic
            result_saliency_maps = []
            original_coords = set(original_saliency_maps.values_list('coordinate', flat=True))
            
            for coordinate in original_coords:
                if coordinate in work_saliency_dict:
                    result_saliency_maps.append(work_saliency_dict[coordinate])
                else:
                    result_saliency_maps.append(original_saliency_maps.get(coordinate=coordinate))
            
            saliency_maps = result_saliency_maps
            
        except WorkGraph.DoesNotExist:
            # Fall back to original saliency maps
            saliency_maps = SaliencyMap.objects.filter(
                input=input_obj,
                coordinate__startswith=f"{layer_name}."
            ).order_by('coordinate')
    else:
        # Get all saliency maps for coordinates that start with the layer name
        saliency_maps = SaliencyMap.objects.filter(
            input=input_obj,
            coordinate__startswith=f"{layer_name}."
        ).order_by('coordinate')
    
    if not saliency_maps:
        raise Http404(f"No saliency maps found for layer '{layer_name}'")
    
    return {
        "layer_name": layer_name,
        "saliency_maps": list(saliency_maps)
    }

@router.post("/models/{model_alias}/inputs/{input_alias}/saliency_maps/batch/", response=LayerSaliencyMapsOut)
def get_batch_saliency_maps(request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn, pruned_graph: str = None):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get saliency maps with pruned graph support
    saliency_maps = get_saliency_maps_with_pruned_graph_support(
        input_obj, 
        data.coordinates, 
        pruned_graph
    )
    
    return {
        "layer_name": "batch",
        "saliency_maps": list(saliency_maps)
    }

@router.get("/models/{model_alias}/inputs/{input_alias}/saliency_maps/single/{coordinate}/", response=SaliencyMapOut)
def get_saliency_map(request, model_alias: str, input_alias: str, coordinate: str, pruned_graph: str = None):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    if pruned_graph:
        # Try to get from WorkSaliencyMap first
        try:
            work_graph = WorkGraph.objects.get(
                work__input=input_obj,
                alias=pruned_graph
            )
            
            try:
                work_saliency_map = WorkSaliencyMap.objects.get(
                    graph=work_graph,
                    coordinate=coordinate
                )
                return work_saliency_map
            except WorkSaliencyMap.DoesNotExist:
                pass  # Fall through to original saliency map
                
        except WorkGraph.DoesNotExist:
            pass  # Fall through to original saliency map
    
    # Default behavior - get original saliency map
    saliency_map = get_object_or_404(
        SaliencyMap,
        input=input_obj,
        coordinate=coordinate
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
            raise Exception(f"only lists or floats are allowed for finding min-max, got={type(items)} items={items}")


    return _find_min_max(data_list)


@router.post("/models/{model_alias}/inputs/{input_alias}/activations/stats/", response=NodeStatsOut)
def get_activations_stats(request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get all activations for the requested coordinates
    activations = list(Activation.objects.filter(
        input=input_obj,
        coordinate__in=data.coordinates
    ).values_list('data', flat=True))
    
    min_val, max_val = get_min_max(activations)
    return {"min": min_val, "max": max_val}


@router.post("/models/{model_alias}/inputs/{input_alias}/saliency_maps/stats/", response=NodeStatsOut)
def get_saliency_maps_stats(request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn, pruned_graph: str = None):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get saliency maps with pruned graph support
    saliency_maps = get_saliency_maps_with_pruned_graph_support(
        input_obj, 
        data.coordinates, 
        pruned_graph
    )
    
    # Extract data from the saliency maps
    saliency_data = [sm.data for sm in saliency_maps]
    
    min_val, max_val = get_min_max(saliency_data)
    return {"min": min_val, "max": max_val}


@router.get("/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/thresholds/", response=list[LayerThresholdOut])
def get_workflow_thresholds(request, model_alias: str, input_alias: str, workflow_name: str):
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    
    # Get all thresholds for this work and model
    thresholds = LayerThreshold.objects.filter(
        work=work,
        model__alias=model_alias
    )
    
    return list(thresholds)


@router.post("/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/thresholds/", response=LayerThresholdOut)
def save_workflow_threshold(request, model_alias: str, input_alias: str, workflow_name: str, data: LayerThresholdIn):
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    model_obj = get_object_or_404(Model, alias=model_alias)
    
    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    
    # Create or update the threshold
    threshold, created = LayerThreshold.objects.update_or_create(
        work=work,
        model=model_obj,
        layer_id=data.layer_id,
        defaults={
            "slider_value": data.slider_value,
            "algorithm": data.algorithm
        }
    )
    
    return threshold


@router.get("/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/pruning/status/")
def get_pruning_status(request, model_alias: str, input_alias: str, workflow_name: str):
    """Get the status of pruning for a workflow - returns completed layers."""
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    
    # Hardcoded pruned graph alias as specified in requirements
    PRUNED_GRAPH_ALIAS = "default_pruned_graph"
    
    try:
        work_graph = WorkGraph.objects.get(
            work=work,
            alias=PRUNED_GRAPH_ALIAS
        )
        
        # Get all completed layers from WorkSaliencyMap
        completed_coordinates = WorkSaliencyMap.objects.filter(
            graph=work_graph
        ).values_list('coordinate', flat=True).distinct()
        
        # Extract layer names from coordinates (e.g., "layers.3.out_0" -> "layers.3")
        completed_layers = set()
        for coord in completed_coordinates:
            if '.' in coord:
                layer_name = '.'.join(coord.split('.')[:2])  # Take first two parts
                completed_layers.add(layer_name)
            elif coord == "inputs":
                completed_layers.add("inputs")
        
        return {
            "completed_layers": list(completed_layers),
            "pruned_graph_alias": PRUNED_GRAPH_ALIAS,
            "total_coordinates_count": len(completed_coordinates)
        }
        
    except WorkGraph.DoesNotExist:
        return {
            "completed_layers": [],
            "pruned_graph_alias": PRUNED_GRAPH_ALIAS,
            "total_coordinates_count": 0
        }


@router.post("/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/pruning/save_layer/")
def save_pruned_layer(request, model_alias: str, input_alias: str, workflow_name: str, data: BatchSaliencyMapsIn):
    """Save the current saliency maps for a layer as WorkSaliencyMap entries."""
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get or create the work/workflow
    work, _ = Work.objects.get_or_create(input=input_obj, name=workflow_name)
    
    # Hardcoded pruned graph alias as specified in requirements
    PRUNED_GRAPH_ALIAS = "default_pruned_graph"
    
    # Get or create the work graph
    work_graph, _ = WorkGraph.objects.get_or_create(
        work=work,
        alias=PRUNED_GRAPH_ALIAS
    )
    
    # Get the current saliency maps for the coordinates
    current_saliency_maps = SaliencyMap.objects.filter(
        input=input_obj,
        coordinate__in=data.coordinates
    )
    
    created_count = 0
    updated_count = 0
    
    # Create or update WorkSaliencyMap entries
    for saliency_map in current_saliency_maps:
        print("updating coordinate", saliency_map.coordinate, work_graph.alias)
        work_saliency_map, created = WorkSaliencyMap.objects.update_or_create(
            graph=work_graph,
            coordinate=saliency_map.coordinate,
            defaults={
                'input': saliency_map.input,
                'data': saliency_map.data,
                'shape': saliency_map.shape,
                'coordinate_type': saliency_map.coordinate_type,
                'data_type': saliency_map.data_type
            }
        )
        
        if created:
            created_count += 1
        else:
            updated_count += 1
    
    return {
        "message": f"Saved {len(data.coordinates)} coordinates to pruned graph",
        "pruned_graph_alias": PRUNED_GRAPH_ALIAS,
        "created_count": created_count,
        "updated_count": updated_count,
        "coordinates_processed": len(current_saliency_maps)
    }
