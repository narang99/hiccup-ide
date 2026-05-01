import pytest
import torch
import torch.nn as nn
import tempfile
import json
import os
from pathlib import Path

from pt_to_api.weights_processor import (
    get_conv_layer_weights_and_biases,
    generate_weights_coordinates,
    process_model_weights_to_coordinates,
    get_weight_coordinate_data,
    save_weights_to_json_files
)
from pt_to_api.persist.weights_processor_outputs import dump_weights_to_files


def test_get_conv_layer_weights_and_biases(test_model):
    """Test extraction of convolutional weights and biases."""
    # Act
    conv_weights, conv_biases = get_conv_layer_weights_and_biases(test_model)
    
    # Assert
    # Should find 2 conv layers: layers.0 and layers.2
    assert len(conv_weights) == 2
    assert "layers.0" in conv_weights
    assert "layers.2" in conv_weights
    
    # Check weight shapes [out_channels, in_channels, kernel_h, kernel_w]
    assert conv_weights["layers.0"].shape == (8, 1, 3, 3)  # Conv2d(1, 8, kernel_size=3)
    assert conv_weights["layers.2"].shape == (16, 8, 3, 3)  # Conv2d(8, 16, kernel_size=3)
    
    # Check biases (model has bias=True by default)
    assert len(conv_biases) == 2
    assert "layers.0" in conv_biases
    assert "layers.2" in conv_biases
    assert conv_biases["layers.0"].shape == (8,)  # 8 output channels
    assert conv_biases["layers.2"].shape == (16,)  # 16 output channels


def test_get_conv_layer_weights_and_biases_no_bias():
    """Test extraction when some layers have no bias."""
    # Arrange - create model with mixed bias settings
    class TestModelNoBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, bias=False),  # No bias
                nn.Conv2d(4, 8, kernel_size=3, bias=True),   # With bias
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModelNoBias()
    
    # Act
    conv_weights, conv_biases = get_conv_layer_weights_and_biases(model)
    
    # Assert
    assert len(conv_weights) == 2
    assert len(conv_biases) == 1  # Only one layer has bias
    assert "layers.0" not in conv_biases
    assert "layers.1" in conv_biases


def test_generate_weights_coordinates(test_model):
    """Test generation of weight coordinates."""
    # Arrange
    conv_weights, conv_biases = get_conv_layer_weights_and_biases(test_model)
    
    # Act
    coordinate_data = generate_weights_coordinates(conv_weights, conv_biases, test_model)
    
    # Assert
    # Check that we have the right number of coordinates
    # layers.0: 8 output channels * 1 input channel = 8 weight coords + 8 bias coords = 16
    # layers.2: 16 output channels * 8 input channels = 128 weight coords + 16 bias coords = 144
    # Total: 16 + 144 = 160
    assert len(coordinate_data) == 160
    
    # Check specific weight coordinates exist
    assert "layers.0.out_0.in_0" in coordinate_data
    assert "layers.0.out_7.in_0" in coordinate_data
    assert "layers.2.out_0.in_0" in coordinate_data
    assert "layers.2.out_15.in_7" in coordinate_data
    
    # Check specific bias coordinates exist
    assert "layers.0.out_0.bias" in coordinate_data
    assert "layers.0.out_7.bias" in coordinate_data
    assert "layers.2.out_0.bias" in coordinate_data
    assert "layers.2.out_15.bias" in coordinate_data


def test_weight_coordinate_data_structure(test_model):
    """Test that weight coordinate data has correct structure."""
    # Arrange
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    # Act
    weight_coord = coordinate_data["layers.0.out_0.in_0"]
    bias_coord = coordinate_data["layers.0.out_0.bias"]
    
    # Assert weight coordinate structure
    assert "data" in weight_coord
    assert "shape" in weight_coord
    assert "layer_type" in weight_coord
    assert "coordinate_type" in weight_coord
    assert "data_type" in weight_coord
    
    assert weight_coord["layer_type"] == "Conv2d"
    assert weight_coord["coordinate_type"] == "input_output_channel"
    assert weight_coord["data_type"] == "weights"
    assert weight_coord["shape"] == [3, 3]  # 3x3 kernel
    assert isinstance(weight_coord["data"], list)
    assert len(weight_coord["data"]) == 3  # 3 rows
    assert len(weight_coord["data"][0]) == 3  # 3 columns
    
    # Assert bias coordinate structure
    assert bias_coord["layer_type"] == "Conv2d"
    assert bias_coord["coordinate_type"] == "output_channel_bias"
    assert bias_coord["data_type"] == "bias"
    assert bias_coord["shape"] == []  # Scalar has empty shape
    assert isinstance(bias_coord["data"], (int, float))  # Single scalar value


def test_weight_coordinate_data_values(test_model):
    """Test that weight coordinate data contains actual weight values."""
    # Arrange
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    # Get the original weight and bias tensors for comparison
    conv_weights, conv_biases = get_conv_layer_weights_and_biases(test_model)
    
    # Act
    weight_coord = coordinate_data["layers.0.out_3.in_0"]
    bias_coord = coordinate_data["layers.0.out_3.bias"]
    
    # Assert
    # Weight data should match the original tensor values
    original_kernel = conv_weights["layers.0"][3, 0]  # out_channel=3, in_channel=0
    expected_weight_data = original_kernel.numpy().tolist()
    assert weight_coord["data"] == expected_weight_data
    
    # Bias data should match the original bias values
    original_bias = conv_biases["layers.0"][3].item()  # out_channel=3
    assert bias_coord["data"] == original_bias


def test_process_model_weights_to_coordinates(test_model):
    """Test the main processing function."""
    # Act
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    # Assert
    assert len(coordinate_data) > 0
    
    # Check that we have coordinates for both conv layers
    layer0_coords = [k for k in coordinate_data.keys() if k.startswith("layers.0")]
    layer2_coords = [k for k in coordinate_data.keys() if k.startswith("layers.2")]
    
    assert len(layer0_coords) == 16  # 8 weight + 8 bias
    assert len(layer2_coords) == 144  # 128 weight + 16 bias
    
    # Check coordinate naming patterns
    weight_coords = [k for k in coordinate_data.keys() if ".bias" not in k]
    bias_coords = [k for k in coordinate_data.keys() if ".bias" in k]
    
    assert len(weight_coords) == 136  # 8 + 128 weight coordinates
    assert len(bias_coords) == 24   # 8 + 16 bias coordinates


def test_get_weight_coordinate_data(test_model):
    """Test getting specific coordinate data."""
    # Arrange
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    # Act
    existing_weight_coord = get_weight_coordinate_data("layers.0.out_0.in_0", coordinate_data)
    existing_bias_coord = get_weight_coordinate_data("layers.0.out_0.bias", coordinate_data)
    missing_coord = get_weight_coordinate_data("non.existent.coord", coordinate_data)
    
    # Assert
    assert existing_weight_coord is not None
    assert "data" in existing_weight_coord
    assert existing_weight_coord["data_type"] == "weights"
    
    assert existing_bias_coord is not None
    assert "data" in existing_bias_coord
    assert existing_bias_coord["data_type"] == "bias"
    
    assert missing_coord is None


def test_save_weights_to_json_files(test_model):
    """Test saving weights to individual JSON files."""
    # Arrange
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Act
        save_weights_to_json_files(coordinate_data, temp_dir)
        
        # Assert
        output_files = list(Path(temp_dir).glob("*.json"))
        assert len(output_files) == len(coordinate_data)
        
        # Check that specific files exist
        expected_files = [
            "layers.0.out_0.in_0.json",
            "layers.0.out_0.bias.json",
            "layers.2.out_15.in_7.json",
            "layers.2.out_15.bias.json"
        ]
        
        for expected_file in expected_files:
            file_path = Path(temp_dir) / expected_file
            assert file_path.exists()
            
            # Check file content is valid JSON with correct structure
            with open(file_path, 'r') as f:
                data = json.load(f)
                assert "data" in data
                assert "shape" in data
                assert "layer_type" in data
                assert "coordinate_type" in data
                assert "data_type" in data


def test_dump_weights_to_files(test_model):
    """Test the persist module function."""
    # Arrange
    coordinate_data = process_model_weights_to_coordinates(test_model)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Act
        dump_weights_to_files(coordinate_data, temp_dir)
        
        # Assert
        output_files = list(Path(temp_dir).glob("*.json"))
        assert len(output_files) == len(coordinate_data)
        
        # Verify file content matches coordinate data
        test_coord = "layers.0.out_0.in_0"
        test_file = Path(temp_dir) / f"{test_coord}.json"
        
        with open(test_file, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data == coordinate_data[test_coord]


def test_weight_coordinate_naming_patterns():
    """Test that coordinate naming follows expected patterns."""
    # Arrange - create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 3, kernel_size=2)  # 2 in, 3 out channels
        
        def forward(self, x):
            return self.conv(x)
    
    model = SimpleModel()
    
    # Act
    coordinate_data = process_model_weights_to_coordinates(model)
    
    # Assert
    weight_coords = sorted([k for k in coordinate_data.keys() if ".bias" not in k])
    bias_coords = sorted([k for k in coordinate_data.keys() if ".bias" in k])
    
    # Check weight coordinate patterns
    expected_weight_coords = [
        "conv.out_0.in_0", "conv.out_0.in_1",
        "conv.out_1.in_0", "conv.out_1.in_1", 
        "conv.out_2.in_0", "conv.out_2.in_1"
    ]
    assert weight_coords == expected_weight_coords
    
    # Check bias coordinate patterns
    expected_bias_coords = [
        "conv.out_0.bias",
        "conv.out_1.bias",
        "conv.out_2.bias"
    ]
    assert bias_coords == expected_bias_coords


def test_weights_different_kernel_sizes():
    """Test weights processing with different kernel sizes."""
    # Arrange - model with different kernel sizes
    class VariedKernelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 2, kernel_size=5)  # 5x5 kernel
            self.conv2 = nn.Conv2d(2, 3, kernel_size=1)  # 1x1 kernel
        
        def forward(self, x):
            return self.conv2(self.conv1(x))
    
    model = VariedKernelModel()
    
    # Act
    coordinate_data = process_model_weights_to_coordinates(model)
    
    # Assert
    # Check that different kernel sizes are handled correctly
    conv1_coord = coordinate_data["conv1.out_0.in_0"]
    conv2_coord = coordinate_data["conv2.out_0.in_0"]
    
    assert conv1_coord["shape"] == [5, 5]  # 5x5 kernel
    assert conv2_coord["shape"] == [1, 1]  # 1x1 kernel
    
    assert len(conv1_coord["data"]) == 5
    assert len(conv1_coord["data"][0]) == 5
    
    assert len(conv2_coord["data"]) == 1
    assert len(conv2_coord["data"][0]) == 1