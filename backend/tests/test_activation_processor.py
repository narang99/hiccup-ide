import pytest
from hiccup_ide.activation_processor import (
    process_activations_to_coordinates,
    get_coordinate_data,
    get_layer_type
)


def test_get_layer_type(model_snapshot):
    """Test layer type identification."""
    # Arrange & Act
    model = model_snapshot['model']
    conv_type = get_layer_type("layers.0", model)
    relu_type = get_layer_type("layers.1", model)
    linear_type = get_layer_type("layers.5", model)
    
    # Assert
    assert conv_type == "Conv2d"
    assert relu_type == "ReLU"
    assert linear_type == "Linear"


def test_activation_capture_includes_input_channels(model_snapshot):
    """Test that activation capture includes input channel coordinates."""
    # Arrange & Act
    activations = model_snapshot['activations']
    
    # Assert
    # Check that we have input channel coordinates for conv layers
    input_channel_coords = [k for k in activations.keys() if ".out_" in k and ".in_" in k]
    assert len(input_channel_coords) > 0
    
    # First conv layer should have coordinates like layers.0.out_X.in_0
    first_conv_coords = [k for k in input_channel_coords if k.startswith("layers.0")]
    assert len(first_conv_coords) == 8  # 8 output channels * 1 input channel
    
    # Second conv layer should have coordinates like layers.2.out_X.in_Y  
    second_conv_coords = [k for k in input_channel_coords if k.startswith("layers.2")]
    assert len(second_conv_coords) == 128  # 16 output channels * 8 input channels


def test_process_activations_to_coordinates(model_snapshot):
    """Test processing activations into coordinate format."""
    # Arrange
    activations = model_snapshot['activations']
    parameters = model_snapshot['parameters']
    model = model_snapshot['model']
    
    # Act
    coordinate_data = process_activations_to_coordinates(activations, parameters, model)
    
    # Assert
    assert len(coordinate_data) > 0
    
    # Test output channel coordinates exist
    assert "layers.0.out_0" in coordinate_data
    assert "layers.1.out_0" in coordinate_data
    assert "layers.2.out_0" in coordinate_data
    
    # Test input channel coordinates exist
    assert "layers.0.out_0.in_0" in coordinate_data
    assert "layers.2.out_0.in_0" in coordinate_data
    assert "layers.2.out_15.in_7" in coordinate_data
    
    # Test linear neuron coordinates exist
    assert "layers.5.out_0" in coordinate_data
    assert "layers.5.out_9" in coordinate_data


def test_coordinate_data_structure(model_snapshot):
    """Test that coordinate data has correct structure."""
    # Arrange
    activations = model_snapshot['activations']
    parameters = model_snapshot['parameters']
    model = model_snapshot['model']
    coordinate_data = process_activations_to_coordinates(activations, parameters, model)
    
    # Act
    conv_output_coord = coordinate_data["layers.0.out_0"]
    conv_input_coord = coordinate_data["layers.0.out_0.in_0"]
    linear_coord = coordinate_data["layers.5.out_0"]
    
    # Assert
    # Conv output channel structure
    assert "data" in conv_output_coord
    assert "shape" in conv_output_coord
    assert "layer_type" in conv_output_coord
    assert "coordinate_type" in conv_output_coord
    assert conv_output_coord["layer_type"] == "Conv2d"
    assert conv_output_coord["coordinate_type"] == "output_channel"
    assert conv_output_coord["shape"] == [14, 14]
    
    # Conv input channel structure
    assert conv_input_coord["layer_type"] == "Conv2d"
    assert conv_input_coord["coordinate_type"] == "input_output_channel"
    assert conv_input_coord["shape"] == [14, 14]
    
    # Linear neuron structure
    assert linear_coord["layer_type"] == "Linear"
    assert linear_coord["coordinate_type"] == "neuron"
    assert linear_coord["shape"] == []
    assert linear_coord["data"] == 0.0


def test_get_coordinate_data(model_snapshot):
    """Test getting specific coordinate data."""
    # Arrange
    activations = model_snapshot['activations']
    parameters = model_snapshot['parameters']
    model = model_snapshot['model']
    coordinate_data = process_activations_to_coordinates(activations, parameters, model)
    
    # Act
    existing_coord = get_coordinate_data("layers.0.out_0", coordinate_data)
    missing_coord = get_coordinate_data("non.existent.coord", coordinate_data)
    
    # Assert
    assert existing_coord is not None
    assert "data" in existing_coord
    assert missing_coord is None


def test_coordinate_count_matches_expected(model_snapshot):
    """Test that we get the expected number of coordinates."""
    # Arrange
    activations = model_snapshot['activations']
    parameters = model_snapshot['parameters']
    model = model_snapshot['model']
    
    # Act
    coordinate_data = process_activations_to_coordinates(activations, parameters, model)
    
    # Assert
    # Expected breakdown:
    # layers.0 (Conv2d): 8 output + 8*1 input = 16 coordinates
    # layers.1 (ReLU): 8 output = 8 coordinates  
    # layers.2 (Conv2d): 16 output + 16*8 input = 144 coordinates
    # layers.3 (ReLU): 16 output = 16 coordinates
    # layers.5 (Linear): 10 neurons = 10 coordinates
    # Total: 16 + 8 + 144 + 16 + 10 = 194 coordinates
    assert len(coordinate_data) == 194