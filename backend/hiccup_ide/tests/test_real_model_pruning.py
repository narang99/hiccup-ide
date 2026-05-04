import pytest
import torch
import os
from django.test import Client
from neural_data.models import Model, Input, Work, WorkGraph, SaliencyMap, WorkSaliencyMap, TempPruneSaliencyMap


# Hardcoded paths matching the saliency.py implementation
MODEL_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/model.pt"
INPUT_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/first-input-tens.pt"


def requires_real_model():
    """Decorator to skip tests if real model files are not available"""
    return pytest.mark.skipif(
        not (os.path.exists(MODEL_PATH) and os.path.exists(INPUT_PATH)),
        reason="Real model and input files not found"
    )


@pytest.fixture
def real_model_data():
    """Create test data that matches the actual model architecture"""
    model = Model.objects.create(
        alias="real-mnist-model",
        name="Real MNIST Model",
        definition={"nodes": [], "edges": []}
    )
    input_obj = Input.objects.create(
        model=model,
        alias="real-mnist-input",
        name="Real MNIST Input",
        data_path=INPUT_PATH
    )
    
    # Load the actual model and input to understand the real shapes
    from pt_to_api.mnist import SimpleMNIST, get_contribs_for_inp_vectorized
    
    actual_model = SimpleMNIST()
    actual_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    actual_model.eval()
    batch_inp_tens = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
    
    # Generate real saliency data using the actual model for layers.3
    dummy_contribs = torch.ones_like(actual_model(batch_inp_tens))  # [batch, 10]
    real_contribs, real_acts, real_params = get_contribs_for_inp_vectorized(
        batch_inp_tens, actual_model, dummy_contribs, "layers.5", "cpu"
    )
    
    # Convert to coordinate format using the actual processor
    from pt_to_api.contrib_processor import process_contribs_to_coordinates
    coord_data = process_contribs_to_coordinates(real_contribs, sample_idx=0)
    
    # Create SaliencyMaps with the real coordinate data
    saliency_maps = []
    for coordinate, data_info in coord_data.items():
        if coordinate.startswith("layers.3."):
            saliency_map = SaliencyMap.objects.create(
                input=input_obj,
                coordinate=coordinate,
                layer_name="layers.3",
                data=data_info["data"],
                shape=data_info["shape"],
                coordinate_type=data_info.get("coordinate_type", "output_channel"),
                data_type="saliency",
                output_channel=data_info.get("output_channel"),
                input_channel=data_info.get("input_channel")
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("layers.2."):
            saliency_map = SaliencyMap.objects.create(
                input=input_obj,
                coordinate=coordinate,
                layer_name="layers.2",
                data=data_info["data"],
                shape=data_info["shape"],
                coordinate_type=data_info.get("coordinate_type", "output_channel"),
                data_type="saliency",
                output_channel=data_info.get("output_channel"),
                input_channel=data_info.get("input_channel")
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("layers.1."):
            saliency_map = SaliencyMap.objects.create(
                input=input_obj,
                coordinate=coordinate,
                layer_name="layers.1",
                data=data_info["data"],
                shape=data_info["shape"],
                coordinate_type=data_info.get("coordinate_type", "output_channel"),
                data_type="saliency",
                output_channel=data_info.get("output_channel"),
                input_channel=data_info.get("input_channel")
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("x."):
            saliency_map = SaliencyMap.objects.create(
                input=input_obj,
                coordinate=coordinate,
                layer_name="x",
                data=data_info["data"],
                shape=data_info["shape"],
                coordinate_type=data_info.get("coordinate_type", "output_channel"),
                data_type="saliency",
                output_channel=data_info.get("output_channel"),
                input_channel=data_info.get("input_channel")
            )
            saliency_maps.append(saliency_map)
    
    return model, input_obj, saliency_maps, real_contribs, batch_inp_tens


def calculate_expected_ground_truth(layer_name, pruned_tensor):
    """Calculate expected ground truth using the actual model"""
    from pt_to_api.mnist import SimpleMNIST, get_contribs_for_inp_vectorized
    
    model = SimpleMNIST()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    batch_inp_tens = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
    
    # Calculate actual contributions
    total_contribs, _, _ = get_contribs_for_inp_vectorized(
        batch_inp_tens, model, pruned_tensor, layer_name, "cpu"
    )
    
    return total_contribs


@pytest.mark.django_db
@requires_real_model()
def test_real_model_layers_3_pruning_ground_truth(real_model_data):
    """Test pruning layers.3 with actual model and validate against ground truth"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()
    
    workflow_name = "real-model-test-3"
    graph_alias = "real-graph-3"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    response = client.post(start_url)
    assert response.status_code == 200
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    # Find a layers.3 coordinate to prune
    layers_3_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.3"]
    assert len(layers_3_maps) > 0, "Need at least one layers.3 coordinate"
    
    target_coordinate = layers_3_maps[0].coordinate
    original_data = layers_3_maps[0].data
    
    # Apply pruning with a threshold that will actually change some values
    # Find a threshold that's between min and max of the original data
    flat_data = torch.tensor(original_data).flatten()
    threshold = float(torch.median(flat_data))
    
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Reconstruct the pruned tensor
    from neural_data.api.saliency import reconstruct_layer_tensor
    pruned_tensor = reconstruct_layer_tensor(graph, "layers.3")
    assert pruned_tensor is not None
    
    # Calculate expected ground truth
    expected_contribs = calculate_expected_ground_truth("layers.3", pruned_tensor)
    
    # Compare with actual propagated values in temp maps
    upstream_layers = ["layers.2", "layers.1", "x"]
    
    for layer_name in upstream_layers:
        if layer_name in expected_contribs:
            # Get the actual propagated values from temp maps
            temp_maps = TempPruneSaliencyMap.objects.filter(
                graph=graph, layer_name=layer_name, coordinate_type="output_channel"
            ).order_by("coordinate")
            
            if temp_maps.exists():
                # Reconstruct actual tensor from temp maps
                actual_tensor = reconstruct_layer_tensor(graph, layer_name)
                expected_tensor = expected_contribs[layer_name]
                
                # Compare shapes
                assert actual_tensor.shape == expected_tensor.shape, \
                    f"Shape mismatch for {layer_name}: actual {actual_tensor.shape} vs expected {expected_tensor.shape}"
                
                # Compare values (with reasonable tolerance for floating point arithmetic)
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, 
                    atol=1e-6, rtol=1e-5,
                    msg=f"Value mismatch for {layer_name}"
                )


@pytest.mark.django_db  
@requires_real_model()
def test_real_model_layers_2_pruning_ground_truth(real_model_data):
    """Test pruning layers.2 with actual model and validate against ground truth"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()
    
    workflow_name = "real-model-test-2"
    graph_alias = "real-graph-2"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    # Find a layers.2 coordinate to prune
    layers_2_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.2"]
    assert len(layers_2_maps) > 0, "Need at least one layers.2 coordinate"
    
    target_coordinate = layers_2_maps[0].coordinate
    original_data = layers_2_maps[0].data
    
    # Apply pruning with a median threshold
    flat_data = torch.tensor(original_data).flatten()
    threshold = float(torch.median(flat_data))
    
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Reconstruct the pruned tensor
    from neural_data.api.saliency import reconstruct_layer_tensor
    pruned_tensor = reconstruct_layer_tensor(graph, "layers.2")
    assert pruned_tensor is not None
    
    # Calculate expected ground truth
    expected_contribs = calculate_expected_ground_truth("layers.2", pruned_tensor)
    
    # Verify upstream layers got correct propagated values
    upstream_layers = ["layers.1", "x"]
    
    for layer_name in upstream_layers:
        if layer_name in expected_contribs:
            temp_maps = TempPruneSaliencyMap.objects.filter(
                graph=graph, layer_name=layer_name, coordinate_type="output_channel"
            ).order_by("coordinate")
            
            if temp_maps.exists():
                actual_tensor = reconstruct_layer_tensor(graph, layer_name)
                expected_tensor = expected_contribs[layer_name]
                
                assert actual_tensor.shape == expected_tensor.shape
                torch.testing.assert_close(
                    actual_tensor, expected_tensor,
                    atol=1e-6, rtol=1e-5,
                    msg=f"Value mismatch for {layer_name}"
                )
    
    # Verify that layers.3 was NOT modified (downstream layer)
    layers_3_modified = TempPruneSaliencyMap.objects.filter(
        graph=graph, layer_name="layers.3", is_modified=True
    )
    assert layers_3_modified.count() == 0, "Downstream layer layers.3 should not be modified"


@pytest.mark.django_db
@requires_real_model()
def test_real_model_input_layer_no_propagation(real_model_data):
    """Test that pruning input layer 'x' with real model doesn't propagate (no upstream layers)"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()
    
    workflow_name = "real-model-input-test"
    graph_alias = "real-input-graph"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    # Find an input layer coordinate
    x_maps = [sm for sm in saliency_maps if sm.layer_name == "x"]
    assert len(x_maps) > 0, "Need at least one x coordinate"
    
    target_coordinate = x_maps[0].coordinate
    original_data = x_maps[0].data
    
    # Store original data of all other layers
    original_temp_data = {}
    for temp_map in TempPruneSaliencyMap.objects.filter(graph=graph).exclude(layer_name="x"):
        original_temp_data[temp_map.coordinate] = temp_map.data.copy()
    
    # Apply pruning to input layer
    flat_data = torch.tensor(original_data).flatten()
    threshold = float(torch.median(flat_data))
    
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Verify the input layer was modified
    x_temp_map = TempPruneSaliencyMap.objects.get(graph=graph, coordinate=target_coordinate)
    assert x_temp_map.is_modified == True
    
    # Verify that no other layers were affected (since x has no upstream)
    for temp_map in TempPruneSaliencyMap.objects.filter(graph=graph).exclude(layer_name="x"):
        assert temp_map.is_modified == False, f"Layer {temp_map.layer_name} should not be user-modified"
        # Data should be unchanged (no propagation occurred)
        assert temp_map.data == original_temp_data[temp_map.coordinate], \
            f"Data for {temp_map.coordinate} should be unchanged"


@pytest.mark.django_db
@requires_real_model()
def test_real_model_algorithm_correctness(real_model_data):
    """Test that algorithms are applied correctly to real model data"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()
    
    workflow_name = "real-algorithm-test"
    graph_alias = "real-algorithm-graph"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    # Test with coordinates from the same layer (API requirement)
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    
    # Test layers.3 coordinates
    layers_3_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.3"][:2]  # Take up to 2
    if layers_3_maps:
        test_cases_3 = []
        for saliency_map in layers_3_maps:
            flat_data = torch.tensor(saliency_map.data).flatten()
            median_val = float(torch.median(flat_data))
            test_cases_3.append({
                "coordinate": saliency_map.coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": median_val},
                "original_data": saliency_map.data
            })
        
        # Calculate expected results and apply pruning for layers.3
        from neural_data.api.helpers import apply_algorithm
        for test_case in test_cases_3:
            test_case["expected_data"] = apply_algorithm(
                test_case["original_data"],
                test_case["algorithm"]
            )
        
        payload_3 = {
            "items": [
                {
                    "coordinate": tc["coordinate"],
                    "algorithm": tc["algorithm"]
                } for tc in test_cases_3
            ]
        }
        response = client.post(save_url, data=payload_3, content_type="application/json")
        assert response.status_code == 200
        
        # Verify each algorithm was applied correctly for layers.3
        for test_case in test_cases_3:
            temp_map = TempPruneSaliencyMap.objects.get(
                graph=graph, coordinate=test_case["coordinate"]
            )
            assert temp_map.is_modified == True
            assert temp_map.data == test_case["expected_data"], \
                f"Algorithm not applied correctly to {test_case['coordinate']}"


@pytest.mark.django_db
@requires_real_model()  
def test_real_model_full_workflow_with_finalization(real_model_data):
    """Test complete real model workflow including finalization"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()
    
    workflow_name = "real-full-workflow"
    graph_alias = "real-full-graph"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    response = client.post(start_url)
    assert response.status_code == 200
    
    # Apply pruning to layers.3
    layers_3_map = next(sm for sm in saliency_maps if sm.layer_name == "layers.3")
    flat_data = torch.tensor(layers_3_map.data).flatten()
    threshold = float(torch.median(flat_data))
    
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": layers_3_map.coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Check status
    status_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/status/"
    status_response = client.get(status_url)
    assert status_response.json()["layers"]["done"] == ["layers.3"]
    assert status_response.json()["session_active"] == True
    
    # Finalize
    finalize_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/finalize_pruning/"
    response = client.post(finalize_url)
    assert response.status_code == 200
    assert response.json()["committed_count"] == 1
    
    # Verify WorkSaliencyMap has the data
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    work_map = WorkSaliencyMap.objects.get(graph=graph, coordinate=layers_3_map.coordinate)
    
    from neural_data.api.helpers import apply_algorithm
    expected_data = apply_algorithm(
        layers_3_map.data,
        {"type": "ThresholdAlgorithm", "threshold": threshold}
    )
    assert work_map.data == expected_data
    
    # Verify temp maps are cleaned up
    assert TempPruneSaliencyMap.objects.filter(graph=graph).count() == 0
    
    # Final status should show no active session
    final_status = client.get(status_url)
    assert final_status.json()["session_active"] == False
    assert final_status.json()["layers"]["done"] == []  # Only checking temp maps now