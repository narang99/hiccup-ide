from ninja import Router
from django.shortcuts import get_object_or_404
from typing import List
from ..models import POI, Work, Weight, Input
from ..schemas import POIIn, POIOut

router = Router()

@router.get("/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/pois/{weight_coordinate}/", response=List[POIOut])
def list_pois(request, model_alias: str, input_alias: str, work_alias: str, weight_coordinate: str):
    pois = POI.objects.filter(
        work__input__model__alias=model_alias,
        work__input__alias=input_alias,
        work__name=work_alias, 
        weight__model__alias=model_alias,
        weight__coordinate=weight_coordinate
    ).select_related('work', 'weight')
    
    return [
        POIOut(
            id=p.pk,
            work_alias=p.work.name,
            weight_coordinate=p.weight.coordinate,
            x=p.x,
            y=p.y,
            label=p.label,
            note=p.note
        ) for p in pois
    ]

@router.post("/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/pois/", response=POIOut)
def create_or_update_poi(request, model_alias: str, input_alias: str, work_alias: str, data: POIIn):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = get_object_or_404(Work, input=input_obj, name=work_alias)
    # The weight must belong to the model this input is using
    weight = get_object_or_404(Weight, model=input_obj.model, coordinate=data.weight_coordinate)
    
    poi, created = POI.objects.update_or_create(
        work=work,
        weight=weight,
        x=data.x,
        y=data.y,
        defaults={
            "label": data.label,
            "note": data.note
        }
    )
    
    return POIOut(
        id=poi.pk,
        work_alias=work.name,
        weight_coordinate=weight.coordinate,
        x=poi.x,
        y=poi.y,
        label=poi.label,
        note=poi.note
    )
