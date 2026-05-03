import pytest
import torch
from pt_to_api.contrib_processor import process_contribs_to_coordinates

def test_process_contribs_to_coordinates_includes_inputs(test_model):
    """Test that contrib processing includes inputs."""
    # Arrange
    # Mock some total_contribs data
    total_contribs = {
        "layers.0": torch.randn(1, 8, 14, 14),
        "x": torch.randn(1, 1, 28, 28),
    }
    
    # Act
    # Without model
    coords_no_model = process_contribs_to_coordinates(total_contribs, 0)
    # With model
    coords_with_model = process_contribs_to_coordinates(total_contribs, 0, test_model)
    
    # Assert
    assert "x.out_0" in coords_no_model
    assert "x.out_0" in coords_with_model
    
    # Check layer_type
    assert coords_no_model["x.out_0"]["layer_type"] == "Input"
    assert coords_with_model["x.out_0"]["layer_type"] == "Input"
    
    # Check other layer_type when model is provided
    assert "layers.0.out_0" in coords_with_model
    assert coords_with_model["layers.0.out_0"]["layer_type"] == "Conv2d"
    
    # Check data_type
    assert coords_with_model["x.out_0"]["data_type"] == "contrib"

def test_process_contribs_to_coordinates_handles_slices():
    """Test that contrib processing handles slices (which always have layer_type Conv2d)."""
    # Arrange
    total_contribs = {
        "layers.0.slice": torch.randn(1, 8, 1, 3, 3),
    }
    
    # Act
    coords = process_contribs_to_coordinates(total_contribs, 0)
    
    # Assert
    assert "layers.0.out_0.in_0" in coords
    assert coords["layers.0.out_0.in_0"]["layer_type"] == "Conv2d"
    assert coords["layers.0.out_0.in_0"]["data_type"] == "contrib"
