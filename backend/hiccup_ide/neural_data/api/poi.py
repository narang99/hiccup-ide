from ninja import Router
from django.shortcuts import get_object_or_404
from typing import List, Optional
from ..models import POI, Work, Weight, Input, Model
from ..schemas import POIIn, POIOut, HighActivatedPOIsResponse
from .high_activated_pois import get_high_activated_pois_for_slice_coordinate

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
            note=p.note,
            larger_pattern=p.larger_pattern
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
            "note": data.note,
            "larger_pattern": data.larger_pattern
        }
    )
    
    return POIOut(
        id=poi.pk,
        work_alias=work.name,
        weight_coordinate=weight.coordinate,
        x=poi.x,
        y=poi.y,
        label=poi.label,
        note=poi.note,
        larger_pattern=poi.larger_pattern
    )


@router.get("/models/{model_alias}/categories/", response=List[str])
def get_unique_categories(request, model_alias: str):
    """
    Get all unique categories for inputs of a given model.
    """
    model = get_object_or_404(Model, alias=model_alias)
    categories = (
        Input.objects.filter(model=model)
        .values_list("category", flat=True)
        .distinct()
        .order_by("category")
    )
    return list(categories)

@router.get("/models/{model_alias}/coordinates/{coordinate}/high_pois/", response=HighActivatedPOIsResponse)
def get_high_activated_pois_for_slice_coordinate_endpoint(
    request,
    model_alias: str,
    coordinate: str,
    k: int = 10,
    categories: Optional[str] = None
):
    """
    Get high-activated POIs for a slice coordinate.
    
    This endpoint finds the highest K contributions from saliency maps across all inputs
    for the same coordinate, then returns the corresponding input/output activations
    and grid coordinates.
    
    Args:
        categories: Comma-separated list of categories to filter by (e.g., "1,2,3")
    """
    category_list = None
    if categories:
        category_list = [cat.strip() for cat in categories.split(',') if cat.strip()]
    
    return get_high_activated_pois_for_slice_coordinate(
        model_alias=model_alias,
        coordinate=coordinate,
        k=k,
        categories=category_list
    )
