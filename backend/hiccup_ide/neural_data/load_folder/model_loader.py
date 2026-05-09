from typing import assert_never
from pathlib import Path
from django.core.files import File
from neural_data.models import Model, Weight
from neural_data.model_plugs import mnist
from neural_data.model_spec import (
    ModelDefinition,
    ModelEdge,
    InputNode,
    InputParams,
    Conv2dNode,
    Conv2dParams,
    ReLUNode,
    ReLUParams,
    FlattenNode,
    FlattenParams,
    LinearNode,
    LinearParams,
    OutputNode,
    OutputParams,
)

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

        # Adapt and validate schema
        validated_schema = adapt_and_validate_model_schema(model_schema)

        model_obj = _create_or_update_model(model_name, validated_schema, model_pt_path)
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

def adapt_and_validate_model_schema(raw_schema):
    """
    Adapts the raw schema from pt-to-api to the strict format expected by model_spec.py
    and validates it.
    """
    nodes = raw_schema.get("nodes", [])
    raw_edges = raw_schema.get("edges", [])
    
    # Map of node id to its raw data
    node_map = {node["id"]: node for node in nodes}
    
    # Map of node id to its incoming source node ids
    incoming_map = {}
    for edge in raw_edges:
        target = edge["target"]
        source = edge["source"]
        if target not in incoming_map:
            incoming_map[target] = []
        incoming_map[target].append(source)
    
    adapted_nodes = []
    
    for node in nodes:
        node_id = node["id"]
        raw_type = node["type"]
        raw_params = node.get("params", {})
        raw_shape = node.get("shape", [])
        
        # Determine strict type
        strict_type = raw_type
        if raw_type == "relu":
            strict_type = "ReLU"
        elif raw_type == "flatten":
            strict_type = "Flatten"
        
        # Determine input shape
        input_shape = []
        sources = incoming_map.get(node_id, [])
        if sources:
            # For simplicity, assume the first source provides the input shape
            source_node = node_map.get(sources[0])
            if source_node:
                input_shape = source_node.get("shape", [])
        
        # Build strict nodes
        if strict_type == "Input":
            adapted_node = InputNode(
                id=node_id,
                shape=raw_shape,
                type="Input",
                params=InputParams(output_shape=raw_shape)
            )
        elif strict_type == "Conv2d":
            adapted_node = Conv2dNode(
                id=node_id,
                shape=raw_shape,
                type="Conv2d",
                params=Conv2dParams(
                    in_channels=raw_params.get("in_channels"),
                    out_channels=raw_params.get("out_channels"),
                    kernel_size=_as_list(raw_params.get("kernel_size", [0, 0])),
                    stride=_as_list(raw_params.get("stride", [1, 1])),
                    padding=_as_list(raw_params.get("padding", [0, 0])),
                    input_shape=input_shape,
                    output_shape=raw_shape,
                )
            )
        elif strict_type == "ReLU":
            adapted_node = ReLUNode(
                id=node_id,
                shape=raw_shape,
                type="ReLU",
                params=ReLUParams(input_shape=input_shape, output_shape=raw_shape)
            )
        elif strict_type == "Flatten":
            adapted_node = FlattenNode(
                id=node_id,
                shape=raw_shape,
                type="Flatten",
                params=FlattenParams(input_shape=input_shape, output_shape=raw_shape)
            )
        elif strict_type == "Linear":
            adapted_node = LinearNode(
                id=node_id,
                shape=raw_shape,
                type="Linear",
                params=LinearParams(
                    in_features=raw_params.get("in_features", input_shape[-1] if input_shape else None),
                    out_features=raw_params.get("out_features", raw_shape[-1] if raw_shape else None),
                    input_shape=input_shape,
                    output_shape=raw_shape,
                )
            )
        elif strict_type == "Output":
            adapted_node = OutputNode(
                id=node_id,
                shape=raw_shape,
                type="Output",
                params=OutputParams(input_shape=input_shape)
            )
        else:
            raise Exception(f"stict_type unknown={strict_type}")
            
        adapted_nodes.append(adapted_node)
        
    edges = [ModelEdge(source=e["source"], target=e["target"]) for e in raw_edges]
    
    # Create complete definition
    model_definition = ModelDefinition(nodes=adapted_nodes, edges=edges)
    return model_definition.model_dump(mode="json")

def _as_list(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val, val]

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
