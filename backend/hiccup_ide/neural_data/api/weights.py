from ninja import Router
from django.shortcuts import get_object_or_404
from typing import Optional
from ..models import Weight
from ..schemas import WeightOut

router = Router()

@router.get("/models/{model_alias}/weights/single/{coordinate}/", response=WeightOut)
def get_weight(
    request, 
    model_alias: str, 
    coordinate: str,
    # Standard query parameters for all data types, not currently used for weights
    work_alias: Optional[str] = None,
    graph_alias: Optional[str] = None,
):
    weight = get_object_or_404(Weight, model__alias=model_alias, coordinate=coordinate)
    return weight
