import torch
import json
import numpy as np
from typing import Dict, Any, Optional


def get_conv_layer_weights_and_biases(model) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract convolutional layer weights and biases from the model.
    Returns tuple of (weights_dict, biases_dict) mapping layer names to their tensors.
    """
    conv_weights = {}
    conv_biases = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Conv2d weights have shape [out_channels, in_channels, kernel_h, kernel_w]
            conv_weights[name] = module.weight.data.clone()
            # Conv2d biases have shape [out_channels] (if bias=True)
            if module.bias is not None:
                conv_biases[name] = module.bias.data.clone()
    
    return conv_weights, conv_biases


def generate_weights_coordinates(conv_weights: Dict[str, torch.Tensor], conv_biases: Dict[str, torch.Tensor], model) -> Dict[str, Dict]:
    """
    Convert convolutional weights and biases to coordinate-indexed dictionaries.
    
    For Conv2d layers with weights shape [out_channels, in_channels, kernel_h, kernel_w]:
    Creates coordinates: 
    - {layer_name}.out_{out_channel}.in_{in_channel} (for kernel weights)
    - {layer_name}.out_{out_channel}.bias (for bias values)
    
    Each coordinate contains the kernel weights for that specific 
    output channel <-> input channel connection, or bias for that output channel.
    
    Args:
        conv_weights: Dict mapping layer names to weight tensors
        conv_biases: Dict mapping layer names to bias tensors
        model: PyTorch model (for consistency with other processors)
        
    Returns:
        Dict mapping coordinate strings to weight/bias data dictionaries
    """
    coordinate_data = {}
    
    for layer_name, weight_tensor in conv_weights.items():
        if len(weight_tensor.shape) != 4:
            # Skip non-Conv2d weights (shouldn't happen given our filtering above)
            continue
            
        out_channels, in_channels, kernel_h, kernel_w = weight_tensor.shape
        
        # Create coordinate for each input-output channel pair (kernel weights)
        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                coord = f"{layer_name}.out_{out_ch}.in_{in_ch}"
                
                # Extract the kernel for this specific channel pair
                kernel_data = weight_tensor[out_ch, in_ch]  # Shape: [kernel_h, kernel_w]
                
                coordinate_data[coord] = {
                    "layer_name": layer_name,
                    "data": kernel_data.numpy().tolist(),
                    "shape": [kernel_h, kernel_w],
                    "layer_type": "Conv2d",
                    "coordinate_type": "input_output_channel",
                    "data_type": "weights"
                }
            
            # Create coordinate for bias (if exists)
            if layer_name in conv_biases:
                bias_coord = f"{layer_name}.out_{out_ch}.bias"
                bias_value = conv_biases[layer_name][out_ch].item()  # Single scalar value
                
                coordinate_data[bias_coord] = {
                    "layer_name": layer_name,
                    "data": bias_value,
                    "shape": [],  # Scalar has no shape
                    "layer_type": "Conv2d",
                    "coordinate_type": "output_channel_bias",
                    "data_type": "bias"
                }
    
    return coordinate_data


def process_model_weights_to_coordinates(model) -> Dict[str, Dict]:
    """
    Process PyTorch model weights and biases and convert them to coordinate-indexed format.
    
    This function extracts convolutional layer weights and biases from a PyTorch model
    and converts them to the same coordinate system used by activations and
    contributions processors.
    
    Args:
        model: PyTorch model to extract weights from
        
    Returns:
        Dictionary where keys are coordinates and values contain:
        For weights:
        - data: The kernel weight values (nested lists for 2D kernels)
        - shape: Shape of the kernel [height, width]
        - layer_type: "Conv2d"
        - coordinate_type: "input_output_channel"
        - data_type: "weights"
        
        For biases:
        - data: The bias value (scalar)
        - shape: [] (empty for scalars)
        - layer_type: "Conv2d"
        - coordinate_type: "output_channel_bias"
        - data_type: "bias"
    """
    conv_weights, conv_biases = get_conv_layer_weights_and_biases(model)
    return generate_weights_coordinates(conv_weights, conv_biases, model)


def get_weight_coordinate_data(coordinate: str, coordinate_data: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
    """
    Get weight or bias data for a specific coordinate.
    Returns the data structure for that coordinate or None if not found.
    
    Args:
        coordinate: Coordinate string like "layers.0.out_2.in_1" or "layers.0.out_2.bias"
        coordinate_data: Dict from process_model_weights_to_coordinates
        
    Returns:
        Dict containing weight/bias data or None if coordinate not found
    """
    return coordinate_data.get(coordinate)


def save_weights_to_json_files(coordinate_data: Dict[str, Dict], output_dir: str = "./weights") -> None:
    """
    Save weight coordinate data to individual JSON files.
    
    Creates one JSON file per coordinate in the specified output directory.
    File naming follows the same pattern as activations:
    {coordinate}.json (e.g., "layers.0.out_0.in_0.json")
    
    Args:
        coordinate_data: Dict from process_model_weights_to_coordinates
        output_dir: Directory to save JSON files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for coord, data in coordinate_data.items():
        filename = f"{coord}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved: {filepath}")