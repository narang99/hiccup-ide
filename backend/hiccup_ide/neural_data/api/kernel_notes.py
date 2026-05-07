from ninja import Router
from django.shortcuts import get_object_or_404

from ..models import KernelNote
from ..schemas import KernelNoteIn, KernelNoteOut

router = Router()


@router.get("/{weight_coordinate}/", response=KernelNoteOut)
def get_kernel_note(request, weight_coordinate: str):
    note, created = KernelNote.objects.get_or_create(
        weight_coordinate=weight_coordinate,
        defaults={'notes': ''}
    )
    
    return {
        "id": note.pk,
        "weight_coordinate": weight_coordinate,
        "notes": note.notes,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
    }


@router.post("/{weight_coordinate}/", response=KernelNoteOut)
def update_kernel_note(request, weight_coordinate: str, data: KernelNoteIn):
    note, created = KernelNote.objects.get_or_create(
        weight_coordinate=weight_coordinate,
        defaults={'notes': data.notes}
    )
    
    if not created:
        note.notes = data.notes
        note.save()
    
    return {
        "id": note.pk,
        "weight_coordinate": weight_coordinate,
        "notes": note.notes,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
    }


@router.put("/{weight_coordinate}/", response=KernelNoteOut)
def replace_kernel_note(request, weight_coordinate: str, data: KernelNoteIn):
    note, created = KernelNote.objects.get_or_create(
        weight_coordinate=weight_coordinate,
        defaults={'notes': data.notes}
    )
    
    if not created:
        note.notes = data.notes
        note.save()
    
    return {
        "id": note.pk,
        "weight_coordinate": weight_coordinate,
        "notes": note.notes,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
    }