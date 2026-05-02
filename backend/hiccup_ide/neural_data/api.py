from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404

from .models import Model, Input, Activation, SaliencyMap, Weight
from .schemas import (
    ModelOut, InputOut, ActivationOut, SaliencyMapOut, 
    WeightOut, LayerSaliencyMapsOut, BatchSaliencyMapsIn
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
def get_layer_saliency_maps(request, model_alias: str, input_alias: str, layer_name: str):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get all saliency maps for coordinates that start with the layer name
    saliency_maps = SaliencyMap.objects.filter(
        input=input_obj,
        coordinate__startswith=f"{layer_name}."
    ).order_by('coordinate')
    
    if not saliency_maps.exists():
        raise Http404(f"No saliency maps found for layer '{layer_name}'")
    
    return {
        "layer_name": layer_name,
        "saliency_maps": list(saliency_maps)
    }

@router.post("/models/{model_alias}/inputs/{input_alias}/saliency_maps/batch/", response=LayerSaliencyMapsOut)
def get_batch_saliency_maps(request, model_alias: str, input_alias: str, data: BatchSaliencyMapsIn):
    # Verify input exists
    input_obj = get_object_or_404(
        Input, 
        model__alias=model_alias, 
        alias=input_alias
    )
    
    # Get all saliency maps for the requested coordinates
    saliency_maps = SaliencyMap.objects.filter(
        input=input_obj,
        coordinate__in=data.coordinates
    ).order_by('coordinate')
    
    return {
        "layer_name": "batch",
        "saliency_maps": list(saliency_maps)
    }

@router.get("/models/{model_alias}/inputs/{input_alias}/saliency_maps/single/{coordinate}/", response=SaliencyMapOut)
def get_saliency_map(request, model_alias: str, input_alias: str, coordinate: str):
    saliency_map = get_object_or_404(
        SaliencyMap,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate
    )
    return saliency_map
