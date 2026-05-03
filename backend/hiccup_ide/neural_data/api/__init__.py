from ninja import Router
from django.shortcuts import get_object_or_404
from ..models import Model, Input
from ..schemas import ModelOut, InputOut
from .activations import router as activations_router
from .weights import router as weights_router
from .saliency import router as saliency_router

router = Router()

# Include sub-routers
router.add_router("", activations_router)
router.add_router("", weights_router)
router.add_router("", saliency_router)


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
