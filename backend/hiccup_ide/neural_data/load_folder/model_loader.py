from pathlib import Path
from django.core.files import File
from neural_data.models import Model, Weight
from neural_data.model_plugs import mnist

def load_model(model_info, folder_path, stdout):
    model_name = model_info.get("name")
    model_path_rel = model_info.get("path")

    if not model_name:
        raise ValueError("Model 'name' is required in meta.json")

    if model_path_rel:
        model_pt_path = folder_path / model_path_rel
        if not model_pt_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_pt_path}")

        stdout.write(f"Processing model: {model_name} from {model_pt_path}")
        model_schema, weight_coordinates = mnist.get_processed_model(str(model_pt_path))

        model_obj = _create_or_update_model(model_name, model_schema, model_pt_path)
        _load_weights(model_obj, weight_coordinates, stdout)
        return model_obj, model_pt_path
    
    # Use existing model
    try:
        model_obj = Model.objects.get(alias=model_name)
        stdout.write(f"Using existing model: {model_name}")
        if not model_obj.pt_file:
            raise ValueError(f"Model {model_name} exists but has no pt_file to process inputs")
        return model_obj, Path(model_obj.pt_file.path)
    except Model.DoesNotExist:
        raise ValueError(f"Model {model_name} not found and no path provided to create it")

def _create_or_update_model(model_name, model_schema, model_pt_path):
    model_obj, created = Model.objects.get_or_create(
        alias=model_name,
        defaults={"name": model_name, "definition": model_schema},
    )

    if not created:
        model_obj.definition = model_schema
        model_obj.name = model_name
    
    with open(model_pt_path, "rb") as f:
        model_obj.pt_file.save(model_pt_path.name, File(f), save=False)
    
    model_obj.save()
    return model_obj

def _load_weights(model_obj, weight_coordinates, stdout):
    stdout.write(f"Loading weights for model {model_obj.alias}...")
    weight_count = 0
    for coord, weight_data in weight_coordinates.items():
        Weight.objects.update_or_create(
            model=model_obj,
            coordinate=coord,
            defaults={
                "layer_name": weight_data.get("layer_name"),
                "data": weight_data["data"],
                "shape": weight_data["shape"],
                "layer_type": weight_data["layer_type"],
                "coordinate_type": weight_data["coordinate_type"],
                "data_type": weight_data["data_type"],
                "output_channel": weight_data.get("output_channel"),
                "input_channel": weight_data.get("input_channel"),
            }
        )
        weight_count += 1
    stdout.write(f"Loaded {weight_count} weights")
