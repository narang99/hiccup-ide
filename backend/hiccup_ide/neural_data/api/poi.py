from ninja import Router
from django.shortcuts import get_object_or_404
from typing import List
from ..models import POI, Work, Weight
from ..schemas import POIIn, POIOut

router = Router()

@router.get("/{work_alias}/{weight_coordinate}/", response=List[POIOut])
def list_pois(request, work_alias: str, weight_coordinate: str):
    pois = POI.objects.filter(
        work__name=work_alias, 
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

@router.post("/", response=POIOut)
def create_or_update_poi(request, data: POIIn):
    work = get_object_or_404(Work, name=data.work_alias)
    weight = get_object_or_404(Weight, coordinate=data.weight_coordinate)
    
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
