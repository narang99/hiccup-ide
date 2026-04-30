import torch
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional


def get_layer_type(layer_name: str, model) -> str:
    """Determine the layer type from the model structure."""
    for name, module in model.named_modules():
        if name == layer_name:
            return module.__class__.__name__
    return "Unknown"


def generate_coordinates(activations: Dict[str, torch.Tensor], parameters: Dict[str, Dict], model) -> Dict[str, Dict]:
    """
    Convert activation tensors to coordinate-indexed dictionaries.
    Input channel coordinates are already captured by the hooks.
    
    Coordinates:
    - Conv layers: {layer_name}.out_{out_channel}.in_{in_channel} and {layer_name}.out_{out_channel}
    - ReLU layers: {layer_name}.out_{out_channel}
    - Linear layers: {layer_name}.out_{out_neuron} (returns zeros for now)
    """
    coordinate_data = {}
    
    for layer_name, activation in activations.items():
        if layer_name == "" or layer_name == "layers":  # Skip empty and top-level layer names
            continue
        
        # Check if this is an input channel coordinate (already computed by hooks)
        if ".out_" in layer_name and ".in_" in layer_name:
            # This is already a coordinate like "layers.0.out_3.in_1"
            # Remove batch dimension if present
            if len(activation.shape) >= 2 and activation.shape[0] == 1:
                activation = activation.squeeze(0)
            
            coordinate_data[layer_name] = {
                "data": activation[0].numpy().tolist() if len(activation.shape) > 2 else activation.numpy().tolist(),
                "shape": list(activation.shape[-2:]) if len(activation.shape) > 2 else list(activation.shape),
                "layer_type": "Conv2d",  # Input channel coords are always from conv layers
                "coordinate_type": "input_output_channel"
            }
            continue
            
        layer_type = get_layer_type(layer_name, model)
        
        # Remove batch dimension if present
        if len(activation.shape) >= 2 and activation.shape[0] == 1:
            activation = activation.squeeze(0)
        
        if layer_type in ["Conv2d", "ReLU"]:
            if len(activation.shape) >= 3:  # [channels, height, width]
                channels, height, width = activation.shape[0], activation.shape[1], activation.shape[2]
                
                # For each output channel
                for out_ch in range(channels):
                    # Main view coordinate: individual output channel
                    out_coord = f"{layer_name}.out_{out_ch}"
                    coordinate_data[out_coord] = {
                        "data": activation[out_ch].numpy().tolist(),
                        "shape": [height, width],
                        "layer_type": layer_type,
                        "coordinate_type": "output_channel"
                    }
            
        elif layer_type == "Linear":
            if len(activation.shape) >= 1:  # [neurons]
                neurons = activation.shape[0]
                
                # For now, return zeros for linear layers
                for neuron in range(neurons):
                    coord = f"{layer_name}.out_{neuron}"
                    coordinate_data[coord] = {
                        "data": 0.0,  # Single value for linear neurons
                        "shape": [],
                        "layer_type": layer_type,
                        "coordinate_type": "neuron"
                    }
            
        elif layer_type == "Flatten":
            # Skip flatten layers for now as they don't have meaningful spatial structure
            continue
    
    return coordinate_data


def process_activations_to_coordinates(activations: Dict[str, torch.Tensor], 
                                     parameters: Dict[str, Dict],
                                     model) -> Dict[str, Dict]:
    """
    Process activation tensors and convert them to coordinate-indexed format.
    
    Returns a dictionary where keys are coordinates and values contain:
    - data: The actual activation values (nested lists for 2D, single values for 1D)
    - shape: Original shape of the data
    - layer_type: Type of the layer (Conv2d, ReLU, Linear)
    - coordinate_type: Type of coordinate (output_channel, input_output_channel, neuron)
    """
    return generate_coordinates(activations, parameters, model)


def save_coordinates_to_json(coordinate_data: Dict[str, Dict], output_path: str):
    """Save coordinate data to JSON file for frontend consumption."""
    with open(output_path, 'w') as f:
        json.dump(coordinate_data, f, indent=2)


def get_coordinate_data(coordinate: str, coordinate_data: Dict[str, Dict]) -> Optional[dict[str, Any]]:
    """
    Get activation data for a specific coordinate.
    Returns the data structure for that coordinate or None if not found.
    """
    return coordinate_data.get(coordinate)