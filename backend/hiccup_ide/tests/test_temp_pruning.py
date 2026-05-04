import pytest
import torch
from django.test import Client
from neural_data.models import Model, Input, Work, WorkGraph, SaliencyMap, WorkSaliencyMap, TempPruneSaliencyMap
from neural_data.api.helpers import apply_algorithm


@pytest.fixture
def sample_data():
    model = Model.objects.create(
        alias="test-model",
        name="Test Model",
        definition={"nodes": [], "edges": []}
    )
    input_obj = Input.objects.create(
        model=model,
        alias="test-input",
        name="Test Input",
        data_path="/test"
    )
    
    # Create realistic saliency maps with proper coordinate types and channels
    # Layer 3 - output channels 0, 1
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.3.out_0",
        layer_name="layers.3",
        data=[[1.0, 2.0], [0.5, 1.5]],  # [H, W] = [2, 2]
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="saliency",
        output_channel=0
    )
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.3.out_1", 
        layer_name="layers.3",
        data=[[3.0, 4.0], [2.5, 3.5]],  # [H, W] = [2, 2]
        shape=[2, 2],
        coordinate_type="output_channel", 
        data_type="saliency",
        output_channel=1
    )
    
    # Layer 2 - output channels 0, 1
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.2.out_0",
        layer_name="layers.2",
        data=[[5.0, 6.0], [4.5, 5.5]],
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="saliency",
        output_channel=0
    )
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.2.out_1",
        layer_name="layers.2", 
        data=[[7.0, 8.0], [6.5, 7.5]],
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="saliency",
        output_channel=1
    )
    
    # Add some upstream layers for ground truth testing
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.1.out_0",
        layer_name="layers.1",
        data=[[9.0, 10.0], [8.5, 9.5]], 
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="saliency",
        output_channel=0
    )
    
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="x.out_0",
        layer_name="x",
        data=[[11.0, 12.0], [10.5, 11.5]],
        shape=[2, 2], 
        coordinate_type="output_channel",
        data_type="saliency",
        output_channel=0
    )
    
    return model, input_obj


def calculate_ground_truth_contribs(last_layer_tensor, last_layer_name):
    """Calculate ground truth contributions using get_contribs_for_inp_vectorized"""
    try:
        # Import the actual functions for ground truth calculation
        from pt_to_api.mnist import get_contribs_for_inp_vectorized, SimpleMNIST
        import os
        
        # Use the cached model and input if available
        MODEL_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/model.pt"
        INPUT_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/first-input-tens.pt"
        
        if os.path.exists(MODEL_PATH) and os.path.exists(INPUT_PATH):
            model = SimpleMNIST()
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            batch_inp_tens = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
            
            # Calculate actual contributions
            total_contribs, _, _ = get_contribs_for_inp_vectorized(
                batch_inp_tens, model, last_layer_tensor, last_layer_name, "cpu"
            )
            
            return total_contribs
        else:
            # Fall back to mock data if files not available
            return create_mock_contribs(last_layer_tensor, last_layer_name)
            
    except Exception:
        # If anything fails, use mock data
        return create_mock_contribs(last_layer_tensor, last_layer_name)


def create_mock_contribs(last_layer_tensor, last_layer_name):
    """Create mock contributions for testing when actual calculation not available"""
    layer_order = ["layers.3", "layers.2", "layers.1", "layers.0", "x"]
    last_idx = layer_order.index(last_layer_name)
    
    mock_contribs = {}
    mock_contribs[last_layer_name] = last_layer_tensor
    
    # For upstream layers, create mock propagated values
    for i in range(last_idx + 1, len(layer_order)):
        layer = layer_order[i]
        if layer == "x":
            mock_contribs[layer] = torch.tensor([[[[15.0, 16.0], [14.5, 15.5]]]])  # [1, 1, 2, 2]
        else:
            # Mock some reasonable propagated values
            mock_contribs[layer] = torch.tensor([[[[13.0, 14.0], [12.5, 13.5]], [[12.0, 13.0], [11.5, 12.5]]]])  # [1, 2, 2, 2]
    
    return mock_contribs

@pytest.mark.django_db
def test_full_pruning_workflow(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "test-workflow"
    graph_alias = "test-graph"
    
    # 1. Start Pruning Session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    response = client.post(start_url)
    assert response.status_code == 200
    assert response.json()["status"] == "started"
    assert response.json()["cloned_count"] == 6
    
    # Verify Temp maps created
    assert TempPruneSaliencyMap.objects.filter(input=input_obj).count() == 6  # All coordinates cloned
    assert TempPruneSaliencyMap.objects.filter(is_modified=True).count() == 0
    
    # 2. Check Status (Nothing modified yet)
    status_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/status/"
    response = client.get(status_url)
    assert response.json()["layers"]["done"] == []
    
    # 3. Save Pruning Results for layer 3
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": "layers.3.out_0",
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 1.5}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    assert response.json()["updated"] == 1
    
    # Verify Temp map modified
    tm = TempPruneSaliencyMap.objects.get(coordinate="layers.3.out_0")
    assert tm.is_modified is True
    # Values < 1.5 -> 0, values >= 1.5 -> original
    # [[1.0, 2.0], [0.5, 1.5]] -> [[0, 2.0], [0, 1.5]]
    expected_data = [[0, 2.0], [0, 1.5]]
    assert tm.data == expected_data
    
    # Verify WorkSaliencyMap still empty
    assert WorkSaliencyMap.objects.count() == 0
    
    # 4. Check Status (Layer 3 should be done)
    response = client.get(status_url)
    assert response.json()["layers"]["done"] == ["layers.3"]
    
    # 5. Finalize Pruning
    finalize_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/finalize_pruning/"
    response = client.post(finalize_url)
    assert response.status_code == 200
    assert response.json()["status"] == "finalized"
    assert response.json()["committed_count"] == 1 # Only the modified one
    
    # Verify WorkSaliencyMap has the data  
    wm = WorkSaliencyMap.objects.get(coordinate="layers.3.out_0")
    assert wm.data == [[0, 2.0], [0, 1.5]]
    
    # Verify Temp maps deleted
    assert TempPruneSaliencyMap.objects.count() == 0
    
    # 6. Check Final Status 
    # After finalization, temp maps are deleted so no layers should be "done"
    # and session_active should be False
    response = client.get(status_url)
    assert response.json()["layers"]["done"] == []
    assert response.json()["session_active"] is False


@pytest.mark.django_db
def test_propagation_occurs_for_layers_3(sample_data):
    """Test that pruning layers.3 triggers propagation to upstream layers"""
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "propagation-test"
    graph_alias = "test-graph-3"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    # Store original values
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    original_layers_2 = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.2.out_0")
    original_layers_1 = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.1.out_0") 
    original_x = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="x.out_0")
    
    original_data_2 = original_layers_2.data.copy()
    original_data_1 = original_layers_1.data.copy() 
    original_data_x = original_x.data.copy()
    
    # Apply pruning to layers.3
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": "layers.3.out_0", 
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 1.5}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Verify that upstream layers have been modified by re-propagation
    updated_layers_2 = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.2.out_0")
    updated_layers_1 = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.1.out_0")
    updated_x = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="x.out_0")
    
    # The upstream layers should have different data due to propagation
    # (Even though they use mock data, the function should still run and potentially change values)
    # At minimum, we verify the propagation function was called without errors
    assert updated_layers_2.is_modified is False  # Not user-modified, but may have propagated values
    assert updated_layers_1.is_modified is False  # Not user-modified, but may have propagated values
    assert updated_x.is_modified is False  # Not user-modified, but may have propagated values

    assert updated_layers_2.data != original_data_2
    assert updated_layers_1.data != original_data_1
    assert updated_x.data != original_data_x
    
    # Verify the pruned layer is marked as modified
    pruned_layer = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.3.out_0")
    assert pruned_layer.is_modified is True
    expected_data = [[0, 2.0], [0, 1.5]]  # Applied threshold 1.5
    assert pruned_layer.data == expected_data


@pytest.mark.django_db  
def test_propagation_layers_2_only_affects_upstream(sample_data):
    """Test that pruning layers.2 only affects upstream layers, not downstream"""
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "layers-2-test"
    graph_alias = "test-graph-2"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    # Store original layers.3 data to verify it doesn't change
    original_layers_3 = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.3.out_0")
    original_layers_3_data = original_layers_3.data.copy()
    
    # Apply pruning to layers.2 (middle layer)
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": "layers.2.out_0",
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 5.5}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Verify that layers.3 was NOT modified (it's downstream)
    layers_3_after = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.3.out_0")
    assert layers_3_after.is_modified is False, "Downstream layer layers.3 should not be user-modified"
    assert layers_3_after.data == original_layers_3_data, "Downstream layer layers.3 data should not change"
    
    # Verify layers.2 itself was modified
    pruned_layer = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.2.out_0")
    assert pruned_layer.is_modified is True
    
    # Verify upstream layers exist and are not user-modified (but may have propagated values)
    layers_1_temp = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.1.out_0")
    x_temp = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="x.out_0")
    assert layers_1_temp.is_modified is False
    assert x_temp.is_modified is False


@pytest.mark.django_db
def test_no_propagation_for_input_layer(sample_data):
    """Test that pruning the input layer 'x' does not trigger re-propagation"""
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "input-layer-test"
    graph_alias = "test-x"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    # Apply pruning to input layer 'x'
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    payload = {
        "items": [
            {
                "coordinate": "x.out_0",
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 10.75}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Verify the input layer was modified
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    
    x_temp_map = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="x.out_0")
    assert x_temp_map.is_modified is True
    # [[11.0, 12.0], [10.5, 11.5]] with threshold 10.75 -> [[11.0, 12.0], [0, 11.5]]
    expected_data = [[11.0, 12.0], [0, 11.5]]
    assert x_temp_map.data == expected_data
    
    # Verify no other layers were affected (since x has no upstream)
    other_modified = TempPruneSaliencyMap.objects.filter(
        graph=graph, is_modified=True
    ).exclude(coordinate="x.out_0")
    assert other_modified.count() == 0, "No upstream layers should be modified when pruning input layer"


@pytest.mark.django_db
def test_algorithm_applied_to_original_data(sample_data):
    """Test that pruning algorithms are always applied to original SaliencyMap data, not propagated data"""
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "algorithm-test" 
    graph_alias = "algorithm-graph"
    
    # Initialize session
    start_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/start_pruning/"
    client.post(start_url)
    
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    save_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/saliency_maps/"
    
    # Apply pruning to layers.3 only (no complex re-propagation issues)
    payload = {
        "items": [{"coordinate": "layers.3.out_0", "algorithm": {"type": "ThresholdAlgorithm", "threshold": 1.5}}]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    
    # Verify the algorithm was applied to the ORIGINAL SaliencyMap data
    pruned_temp_map = TempPruneSaliencyMap.objects.get(graph=graph, coordinate="layers.3.out_0")
    assert pruned_temp_map.is_modified is True
    
    # Get the original base data and apply algorithm manually
    original_saliency_map = SaliencyMap.objects.get(input=input_obj, coordinate="layers.3.out_0")
    expected_data = apply_algorithm(
        original_saliency_map.data,
        {"type": "ThresholdAlgorithm", "threshold": 1.5}
    )
    
    # The temp map should match the manually applied algorithm result
    assert pruned_temp_map.data == expected_data
    
    # Status should show layers.3 as done
    status_url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/status/"
    status_response = client.get(status_url)
    assert status_response.json()["layers"]["done"] == ["layers.3"]
    assert status_response.json()["session_active"] is True
