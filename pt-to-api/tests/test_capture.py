import pytest
import torch
from pt_to_api.capture import (
    get_model_internals, 
    ModelSnapshot,
    compute_conv_input_channel_contributions
)


def test_get_model_internals_returns_activations_and_parameters(test_model, test_input):
    """Test that get_model_internals returns both activations and parameters."""
    # Act
    activations, parameters = get_model_internals(test_model, test_input)
    
    # Assert
    assert isinstance(activations, dict)
    assert isinstance(parameters, dict)
    assert len(activations) > 0
    assert len(parameters) > 0
    assert "x" in activations
    assert activations["x"].shape == (1, 1, 28, 28)


def test_model_snapshot_captures_layer_outputs(test_model, test_input):
    """Test that ModelSnapshot captures layer outputs correctly."""
    # Arrange
    snapshot = ModelSnapshot(test_model)
    
    # Act
    test_model.eval()
    with torch.no_grad():
        _ = test_model(test_input)
    
    # Assert
    assert len(snapshot.activations) > 0
    assert "layers.0" in snapshot.activations  # First conv layer
    assert "layers.1" in snapshot.activations  # First ReLU layer
    
    # Check shapes are correct
    assert snapshot.activations["layers.0"].shape == (1, 8, 14, 14)
    assert snapshot.activations["layers.1"].shape == (1, 8, 14, 14)
    
    # Cleanup
    snapshot.remove()


def test_model_snapshot_captures_parameters(test_model):
    """Test that ModelSnapshot captures layer parameters correctly."""
    # Act
    snapshot = ModelSnapshot(test_model)
    
    # Assert
    assert "layers.0" in snapshot.parameters
    assert "layers.2" in snapshot.parameters
    assert "layers.5" in snapshot.parameters
    
    # Check conv layer parameters
    conv_params = snapshot.parameters["layers.0"]
    assert "weight" in conv_params
    assert "bias" in conv_params
    assert "stride" in conv_params
    assert "padding" in conv_params
    
    # Check linear layer parameters
    linear_params = snapshot.parameters["layers.5"]
    assert "weight" in linear_params
    assert "bias" in linear_params
    
    # Cleanup
    snapshot.remove()


def test_compute_conv_input_channel_contributions(test_model, test_input):
    """Test computing individual input channel contributions for conv layers."""
    # Arrange
    conv_layer = test_model.layers[0]  # First conv layer
    
    # Act
    contributions = compute_conv_input_channel_contributions(conv_layer, (test_input,))
    
    # Assert
    assert isinstance(contributions, dict)
    assert len(contributions) == 8  # 8 output channels * 1 input channel
    
    # Check coordinate naming
    assert "out_0.in_0" in contributions
    assert "out_7.in_0" in contributions
    
    # Check tensor shapes
    for coord, (contribution, input_slice) in contributions.items():
        assert contribution.shape == (1, 1, 14, 14)  # [batch, 1, H, W]
        assert input_slice.shape == (1, 1, 28, 28)


def test_conv_input_channel_contributions_captured_in_activations(test_model, test_input):
    """Test that input channel contributions are captured during forward pass."""
    # Act
    activations, _ = get_model_internals(test_model, test_input)
    
    # Assert
    # Check that input channel coordinates exist
    input_coords = [k for k in activations.keys() if ".out_" in k and ".in_" in k]
    assert len(input_coords) > 0
    
    # Check specific coordinates exist
    assert "layers.0.out_0.in_0" in activations
    assert "layers.2.out_0.in_0" in activations
    
    # Check tensor shapes for input channel contributions
    assert activations["layers.0.out_0.in_0"].shape == (1, 1, 14, 14)
    assert activations["layers.2.out_0.in_0"].shape == (1, 1, 7, 7)


def test_bias_distribution_in_input_channel_contributions(test_model, test_input):
    """Test that bias is distributed evenly across input channel contributions."""
    # Arrange
    conv_layer = test_model.layers[0]  # First conv layer (1 -> 8 channels)
    
    # Act
    contributions = compute_conv_input_channel_contributions(conv_layer, (test_input,))
    
    # Get the full layer output for comparison
    with torch.no_grad():
        full_output = conv_layer(test_input)  # [1, 8, 14, 14]
    
    # Assert
    # For first conv layer with 1 input channel, each contribution should get full bias
    # since bias[out_ch] / 1 = bias[out_ch]
    for out_ch in range(8):
        coord_key = f"out_{out_ch}.in_0"
        contribution, input_slice = contributions[coord_key]  # [1, 1, 14, 14]
        
        # The contribution should equal the full output for that channel
        # (since there's only 1 input channel, it gets the full bias)
        expected = full_output[:, out_ch:out_ch+1]  # [1, 1, 14, 14]
        
        # Check shapes match
        assert contribution.shape == expected.shape
        
        # Check values are approximately equal (allowing for small numerical differences)
        assert torch.allclose(contribution, expected, atol=1e-6)


def test_bias_distribution_multi_input_channels():
    """Test bias distribution with multiple input channels."""
    # Arrange - Create a simple conv layer with multiple input channels
    conv_layer = torch.nn.Conv2d(3, 2, kernel_size=1)  # 3 input, 2 output channels
    input_tensor = torch.randn(1, 3, 4, 4)
    
    # Act
    contributions = compute_conv_input_channel_contributions(conv_layer, (input_tensor,))
    
    # Assert
    # Should have 6 contributions (2 output * 3 input channels)
    assert len(contributions) == 6
    
    # Check that sum of input channel contributions equals full output
    with torch.no_grad():
        full_output = conv_layer(input_tensor)  # [1, 2, 4, 4]
    
    for out_ch in range(2):
        # Sum contributions for this output channel
        channel_sum = torch.zeros_like(full_output[:, out_ch:out_ch+1])
        for in_ch in range(3):
            coord_key = f"out_{out_ch}.in_{in_ch}"
            channel_sum += contributions[coord_key][0]
        
        # Should equal the full output for that channel
        expected = full_output[:, out_ch:out_ch+1]
        assert torch.allclose(channel_sum, expected, atol=1e-5)