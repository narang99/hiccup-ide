from typing import List
from ninja import Router
from django.shortcuts import get_object_or_404
from ..models import Model, Input, Work, WorkGraph, PinnedWork
from ..schemas import ModelOut, InputOut, ModelTreeOut, InputTreeOut, WorkTreeOut, WorkIn
from .activations import router as activations_router
from .weights import router as weights_router
from .saliency import router as saliency_router
from .poi import router as poi_router
from .kernel_labels import router as kernel_labels_router
from .kernel_notes import router as kernel_notes_router
from .prune import router as prune_router

router = Router()

# Include sub-routers
router.add_router("", activations_router)
router.add_router("", weights_router)
router.add_router("", saliency_router)
router.add_router("", poi_router)
router.add_router("", prune_router)
router.add_router("kernel-labels", kernel_labels_router)
router.add_router("kernel-notes", kernel_notes_router)


@router.get("/")
def root(request):
    return {"message": "Hiccup IDE API"}


@router.get("/workspace/", response=List[ModelTreeOut])
def get_workspace(request):
    models = Model.objects.all().prefetch_related('inputs__works__pinned')
    result = []
    for model in models:
        # Get all inputs for the model
        inputs = Input.objects.filter(model=model).prefetch_related('works__pinned')
        
        # Sort inputs: those with pinned works first
        def input_has_pinned_work(input_obj):
            return any(hasattr(work, 'pinned') for work in input_obj.works.all())
        
        sorted_inputs = sorted(inputs, key=input_has_pinned_work, reverse=True)
        
        model_data = ModelTreeOut(
            alias=model.alias,
            name=model.name,
            inputs=[
                InputTreeOut(
                    alias=input_obj.alias,
                    name=input_obj.name,
                    works=[
                        WorkTreeOut(
                            alias=work.name, 
                            name=work.name,
                            is_pinned=hasattr(work, 'pinned')
                        )
                        for work in sorted(
                            input_obj.works.all(), 
                            key=lambda w: (not hasattr(w, 'pinned'), w.created_at),
                            reverse=False
                        )
                    ]
                )
                for input_obj in sorted_inputs
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
    return WorkTreeOut(alias=work.name, name=work.name, is_pinned=False)


@router.post("/models/{model_alias}/inputs/{input_alias}/works/{work_alias}/pin/")
def pin_work(request, model_alias: str, input_alias: str, work_alias: str):
    work = get_object_or_404(Work, input__model__alias=model_alias, input__alias=input_alias, name=work_alias)
    pinned_work, created = PinnedWork.objects.get_or_create(work=work)
    return {"success": True, "action": "pinned" if created else "already_pinned"}


@router.delete("/models/{model_alias}/inputs/{input_alias}/works/{work_alias}/pin/")
def unpin_work(request, model_alias: str, input_alias: str, work_alias: str):
    work = get_object_or_404(Work, input__model__alias=model_alias, input__alias=input_alias, name=work_alias)
    deleted_count, _ = PinnedWork.objects.filter(work=work).delete()
    return {"success": True, "action": "unpinned" if deleted_count > 0 else "not_pinned"}


@router.get("/models/{model_alias}/pinned-works/", response=List[WorkTreeOut])
def get_pinned_works_for_model(request, model_alias: str):
    pinned_works = PinnedWork.objects.filter(
        work__input__model__alias=model_alias,
        work__graph__isnull=False  # Only include works that have a WorkGraph
    ).select_related('work__input', 'work__graph').order_by('-created_at')
    
    return [
        WorkTreeOut(
            alias=f"{pinned_work.work.input.alias}/{pinned_work.work.name}",
            name=f"{pinned_work.work.input.name} → {pinned_work.work.name}",
            is_pinned=True
        )
        for pinned_work in pinned_works
    ]
