from typing import List
from ninja import Router
from django.shortcuts import get_object_or_404
from ..models import Model, Input, Work, WorkGraph
from ..schemas import ModelOut, InputOut, ModelTreeOut, InputTreeOut, WorkTreeOut, WorkIn
from .activations import router as activations_router
from .weights import router as weights_router
from .saliency import router as saliency_router
from .poi import router as poi_router

router = Router()

# Include sub-routers
router.add_router("", activations_router)
router.add_router("", weights_router)
router.add_router("", saliency_router)
router.add_router("", poi_router)


@router.get("/")
def root(request):
    return {"message": "Hiccup IDE API"}


@router.get("/workspace/", response=List[ModelTreeOut])
def get_workspace(request):
    models = Model.objects.all().prefetch_related('inputs__works')
    result = []
    for model in models:
        model_data = ModelTreeOut(
            alias=model.alias,
            name=model.name,
            inputs=[
                InputTreeOut(
                    alias=input_obj.alias,
                    name=input_obj.name,
                    works=[
                        WorkTreeOut(alias=work.name, name=work.name)
                        for work in Work.objects.filter(input=input_obj)
                    ]
                )
                for input_obj in Input.objects.filter(model=model)
            ]
        )
        result.append(model_data)
    return result


@router.get("/models/{model_alias}/", response=ModelOut)
def get_model(request, model_alias: str):
    model = get_object_or_404(Model, alias=model_alias)
    return model


@router.get("/models/{model_alias}/inputs/{input_alias}/", response=InputOut)
def get_input(request, model_alias: str, input_alias: str):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    return input_obj


@router.post("/models/{model_alias}/inputs/{input_alias}/works/", response=WorkTreeOut)
def create_work(request, model_alias: str, input_alias: str, data: WorkIn):
    input_obj = get_object_or_404(Input, model__alias=model_alias, alias=input_alias)
    work = Work.objects.create(input=input_obj, name=data.name)
    WorkGraph.objects.create(work=work)
    return WorkTreeOut(alias=work.name, name=work.name)
