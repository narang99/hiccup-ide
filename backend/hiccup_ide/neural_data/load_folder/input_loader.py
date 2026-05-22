from pathlib import Path
from tqdm import tqdm
from django.core.files import File
from neural_data.models import Input, Activation, SaliencyMap
from neural_data.model_plugs import mnist

def load_inputs(inputs_list, model_obj, folder_path, model_pt_path, stdout):
    for i, input_info in enumerate(inputs_list):
        _load_single_input(input_info, model_obj, folder_path, model_pt_path, stdout)
        if i % 50 == 0:
            print(f"done: {i}")

def _load_single_input(input_info, model_obj, folder_path, model_pt_path, stdout):
    inp_path_rel = input_info.get("path")
    inp_label = input_info.get("label")

    if not inp_path_rel or inp_label is None:
        stdout.write(f"Skipping invalid input: {input_info}")
        return

    inp_pt_path = folder_path / inp_path_rel
    if not inp_pt_path.exists():
        stdout.write(f"Input file not found: {inp_pt_path}")
        return

    inp_alias = inp_pt_path.stem
    stdout.write(f"Processing input: {inp_alias} (label: {inp_label})")
    if Input.objects.filter(model=model_obj, alias=inp_alias).exists():
        stdout.write(f"WARN: skipping input: {inp_alias}, already exists")
        return

    input_obj = _create_or_update_input(model_obj, inp_alias, inp_path_rel, str(inp_label), inp_pt_path)

    # Process activations and saliency maps
    input_tensor = mnist.load_input_tensor(str(inp_pt_path))
    act_coords, contrib_coords = mnist.get_processed_activations_and_contribs(
        str(model_pt_path), input_tensor, inp_label
    )

    act_count = _load_activations(input_obj, act_coords)
    sal_count = _load_saliency_maps(input_obj, contrib_coords)
    
    stdout.write(f"Loaded {act_count} activations and {sal_count} saliency maps for {inp_alias}")

def _create_or_update_input(model_obj, inp_alias, inp_path_rel, category, inp_pt_path):
    input_obj, created = Input.objects.get_or_create(
        model=model_obj,
        alias=inp_alias,
        defaults={"name": inp_alias, "data_path": str(inp_path_rel), "category": category},
    )
    
    if not created:
        input_obj.data_path = str(inp_path_rel)
        input_obj.name = inp_alias
        input_obj.category = category
    
    with open(inp_pt_path, "rb") as f:
        input_obj.pt_file.save(inp_pt_path.name, File(f), save=False)
    input_obj.save()
    return input_obj

def _load_activations(input_obj, act_coords):
    act_count = 0
    for coord, act_data in act_coords.items():
        Activation.objects.update_or_create(
            input=input_obj,
            coordinate=coord,
            defaults={
                "layer_name": act_data.get("layer_name"),
                "data": act_data["data"],
                "shape": act_data["shape"],
                "layer_type": act_data["layer_type"],
                "coordinate_type": act_data["coordinate_type"],
                "output_channel": act_data.get("output_channel"),
                "input_channel": act_data.get("input_channel"),
            }
        )
        act_count += 1
    return act_count

def _load_saliency_maps(input_obj, contrib_coords):
    sal_count = 0
    for coord, sal_data in contrib_coords.items():
        SaliencyMap.objects.update_or_create(
            input=input_obj,
            coordinate=coord,
            defaults={
                "layer_name": sal_data.get("layer_name"),
                "data": sal_data["data"],
                "shape": sal_data["shape"],
                "coordinate_type": sal_data["coordinate_type"],
                "data_type": sal_data.get("data_type", "contrib"),
                "output_channel": sal_data.get("output_channel"),
                "input_channel": sal_data.get("input_channel"),
            }
        )
        sal_count += 1
    return sal_count
