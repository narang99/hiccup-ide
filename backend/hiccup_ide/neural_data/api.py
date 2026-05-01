from ninja import Router
from django.shortcuts import get_object_or_404
from django.http import Http404

from .models import Model, Input, Activation, SaliencyMap, Weight
from .schemas import ModelOut, InputOut, ActivationOut, SaliencyMapOut, WeightOut

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

@router.get("/models/{model_alias}/inputs/{input_alias}/activations/{coordinate}/", response=ActivationOut)
def get_activation(request, model_alias: str, input_alias: str, coordinate: str):
    activation = get_object_or_404(
        Activation,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate
    )
    return activation

@router.get("/models/{model_alias}/inputs/{input_alias}/saliency_maps/{coordinate}/", response=SaliencyMapOut)
def get_saliency_map(request, model_alias: str, input_alias: str, coordinate: str):
    saliency_map = get_object_or_404(
        SaliencyMap,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate
    )
    return saliency_map

@router.get("/models/{model_alias}/weights/{coordinate}/", response=WeightOut)
def get_weight(request, model_alias: str, coordinate: str):
    weight = get_object_or_404(
        Weight,
        model__alias=model_alias,
        coordinate=coordinate
    )
    return weight