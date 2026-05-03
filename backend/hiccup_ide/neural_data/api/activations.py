from ninja import Router
from django.shortcuts import get_object_or_404
from typing import Optional
from ..models import Activation, Input
from ..schemas import ActivationOut, BatchSaliencyMapsIn, NodeStatsOut
from .helpers import get_min_max

router = Router()

@router.get(
    "/models/{model_alias}/inputs/{input_alias}/activations/single/{coordinate}/",
    response=ActivationOut,
)
def get_activation(
    request, 
    model_alias: str, 
    input_alias: str, 
    coordinate: str,
    # Standard query parameters for all data types, not currently used for activations
    work_alias: Optional[str] = None,
    graph_alias: Optional[str] = None,
):
    activation = get_object_or_404(
        Activation,
        input__model__alias=model_alias,
        input__alias=input_alias,
        coordinate=coordinate,
    )
    return activation

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
