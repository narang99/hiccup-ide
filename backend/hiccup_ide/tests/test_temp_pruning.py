import pytest
from django.test import Client
from neural_data.models import Model, Input, Work, WorkGraph, SaliencyMap, WorkSaliencyMap, TempPruneSaliencyMap

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
    
    # Create some base saliency maps
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.3.0",
        layer_name="layers.3",
        data=[[1.0, 2.0]],
        shape=[1, 2],
        coordinate_type="channel",
        data_type="saliency"
    )
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layers.2.0",
        layer_name="layers.2",
        data=[[3.0, 4.0]],
        shape=[1, 2],
        coordinate_type="channel",
        data_type="saliency"
    )
    
    return model, input_obj

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
    assert response.json()["cloned_count"] == 2
    
    # Verify Temp maps created
    assert TempPruneSaliencyMap.objects.filter(input=input_obj).count() == 2
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
                "coordinate": "layers.3.0",
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 1.5}
            }
        ]
    }
    response = client.post(save_url, data=payload, content_type="application/json")
    assert response.status_code == 200
    assert response.json()["updated"] == 1
    
    # Verify Temp map modified
    tm = TempPruneSaliencyMap.objects.get(coordinate="layers.3.0")
    assert tm.is_modified is True
    # 1.0 < 1.5 -> 0, 2.0 >= 1.5 -> 2.0
    assert tm.data == [[0, 2.0]]
    
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
    wm = WorkSaliencyMap.objects.get(coordinate="layers.3.0")
    assert wm.data == [[0, 2.0]]
    
    # Verify Temp maps deleted
    assert TempPruneSaliencyMap.objects.count() == 0
    
    # 6. Check Final Status
    # Since status exclusively uses TempPruneSaliencyMap, and it was deleted on finalize,
    # the "done" list should now be empty for the next potential session.
    response = client.get(status_url)
    assert response.json()["layers"]["done"] == []
