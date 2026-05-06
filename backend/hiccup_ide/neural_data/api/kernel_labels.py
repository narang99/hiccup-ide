from ninja import Router
from django.shortcuts import get_object_or_404
from typing import List

from ..models import KernelLabels, Weight
from ..schemas import KernelLabelsIn, KernelLabelsOut

router = Router()


@router.get("/{weight_coordinate}/", response=KernelLabelsOut)
def get_kernel_labels(request, weight_coordinate: str):
    weight = get_object_or_404(Weight, coordinate=weight_coordinate)
    labels, created = KernelLabels.objects.get_or_create(
        weight=weight,
        defaults={'labels': ['spurious']}
    )
    
    return {
        "id": labels.pk,
        "weight_coordinate": weight_coordinate,
        "labels": labels.labels,
        "created_at": labels.created_at.isoformat(),
        "updated_at": labels.updated_at.isoformat(),
    }


@router.post("/{weight_coordinate}/", response=KernelLabelsOut)
def update_kernel_labels(request, weight_coordinate: str, data: KernelLabelsIn):
    weight = get_object_or_404(Weight, coordinate=weight_coordinate)
    labels, created = KernelLabels.objects.get_or_create(
        weight=weight,
        defaults={'labels': data.labels}
    )
    
    if not created:
        labels.labels = data.labels
        labels.save()
    
    return {
        "id": labels.pk,
        "weight_coordinate": weight_coordinate,
        "labels": labels.labels,
        "created_at": labels.created_at.isoformat(),
        "updated_at": labels.updated_at.isoformat(),
    }


@router.post("/{weight_coordinate}/add-label/")
def add_label(request, weight_coordinate: str, label: str):
    weight = get_object_or_404(Weight, coordinate=weight_coordinate)
    labels, created = KernelLabels.objects.get_or_create(
        weight=weight,
        defaults={'labels': ['spurious']}
    )
    
    if label not in labels.labels:
        labels.labels.append(label)
        labels.save()
    
    return {"success": True, "labels": labels.labels}


@router.post("/{weight_coordinate}/remove-label/")
def remove_label(request, weight_coordinate: str, label: str):
    weight = get_object_or_404(Weight, coordinate=weight_coordinate)
    labels = get_object_or_404(KernelLabels, weight=weight)
    
    if label in labels.labels and label != 'spurious':
        labels.labels.remove(label)
        labels.save()
    
    return {"success": True, "labels": labels.labels}