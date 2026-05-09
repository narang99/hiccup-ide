import pytest
from neural_data.load_folder.model_loader import adapt_and_validate_model_schema

def test_adapt_and_validate_model_schema():
    raw_schema = {
        "nodes": [
            {"id": "input", "type": "Input", "params": {}, "shape": [1, 1, 28, 28]},
            {"id": "conv1", "type": "Conv2d", "params": {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1}, "shape": [1, 8, 14, 14]},
            {"id": "relu1", "type": "relu", "params": {}, "shape": [1, 8, 14, 14]},
            {"id": "flatten", "type": "flatten", "params": {}, "shape": [1, 1568]},
            {"id": "output", "type": "Output", "params": {}, "shape": [1, 1568]}
        ],
        "edges": [
            {"source": "input", "target": "conv1"},
            {"source": "conv1", "target": "relu1"},
            {"source": "relu1", "target": "flatten"},
            {"source": "flatten", "target": "output"}
        ]
    }

    validated = adapt_and_validate_model_schema(raw_schema)
    
    # Check that types are translated
    types = {node["id"]: node["type"] for node in validated["nodes"]}
    assert types["relu1"] == "ReLU"
    assert types["flatten"] == "Flatten"
    
    # Check that params are populated
    nodes = {node["id"]: node for node in validated["nodes"]}
    
    # Conv2d params
    conv1_params = nodes["conv1"]["params"]
    assert conv1_params["in_channels"] == 1
    assert conv1_params["input_shape"] == [1, 1, 28, 28]
    assert conv1_params["output_shape"] == [1, 8, 14, 14]
    
    # ReLU params
    relu1_params = nodes["relu1"]["params"]
    assert relu1_params["input_shape"] == [1, 8, 14, 14]
    assert relu1_params["output_shape"] == [1, 8, 14, 14]
    
    # Flatten params
    flatten_params = nodes["flatten"]["params"]
    assert flatten_params["input_shape"] == [1, 8, 14, 14]
    assert flatten_params["output_shape"] == [1, 1568]

def test_validation_fails_on_invalid_schema():
    invalid_schema = {
        "nodes": [
            {"id": "input", "type": "Input", "params": {}, "shape": [1, 1, 28, 28]},
            # Missing Conv2d params like in_channels
            {"id": "conv1", "type": "Conv2d", "params": {}, "shape": [1, 8, 14, 14]},
        ],
        "edges": []
    }
    
    # Our adapter currently provides defaults (0), so it might actually pass 
    # unless we make it stricter or check for specific required fields that don't have defaults.
    # In model_spec.py, Conv2dParams has required fields without defaults in Pydantic.
    
    with pytest.raises(Exception):
        adapt_and_validate_model_schema(invalid_schema)
