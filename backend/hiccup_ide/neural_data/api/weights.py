from ninja import Router
from django.shortcuts import get_object_or_404
from ..models import Weight
from ..schemas import WeightOut

router = Router()

@router.get("/models/{model_alias}/weights/single/{coordinate}/", response=WeightOut)
def get_weight(request, model_alias: str, coordinate: str):
    weight = get_object_or_404(Weight, model__alias=model_alias, coordinate=coordinate)
    return weight
