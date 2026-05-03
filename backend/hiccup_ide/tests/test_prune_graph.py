import pytest
from django.test import Client
from neural_data.models import Model, Input, Work, WorkGraph

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
    return model, input_obj

@pytest.mark.django_db
def test_create_work_graph_via_api(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "test-workflow"
    graph_alias = "test-graph"
    
    # Act
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/"
    response = client.post(url)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["alias"] == graph_alias
    assert data["created"] is True
    
    # Verify in DB
    assert WorkGraph.objects.filter(alias=graph_alias, work__name=workflow_name).exists()

@pytest.mark.django_db
def test_update_work_graph_via_api(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "test-workflow"
    graph_alias = "test-graph"
    
    # Create initial
    work = Work.objects.create(input=input_obj, name=workflow_name)
    WorkGraph.objects.create(work=work, alias=graph_alias)
    
    # Act - calling the same endpoint should return existing
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/"
    response = client.post(url)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["alias"] == graph_alias
    assert data["created"] is False

@pytest.mark.django_db
def test_get_workflow_pruning_status_sequential(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    workflow_name = "test-workflow"
    graph_alias = "test-graph"
    
    # Create work and graph
    work = Work.objects.create(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.create(work=work, alias=graph_alias)
    
    # Define layers (TOTAL_LAYERS = ["layers.3", "layers.2", "layers.1", "layers.0", "inputs"])
    # Case 1: No layers done
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/workflows/{workflow_name}/graphs/{graph_alias}/status/"
    response = client.get(url)
    assert response.status_code == 200
    assert response.json()["layers"]["done"] == []
    assert response.json()["layers"]["total"] == ["layers.3", "layers.2", "layers.1", "layers.0", "x"]
    
    # Case 2: Only the first layer done
    from neural_data.models import WorkSaliencyMap
    WorkSaliencyMap.objects.create(
        graph=graph, 
        input=input_obj, 
        coordinate="layers.3.out_0", 
        layer_name="layers.3",
        data={}, shape=[], coordinate_type="", data_type=""
    )
    
    response = client.get(url)
    assert response.json()["layers"]["done"] == ["layers.3"]
    
    # Case 3: Sequential layers done
    WorkSaliencyMap.objects.create(
        graph=graph, 
        input=input_obj, 
        coordinate="layers.2.out_0", 
        layer_name="layers.2",
        data={}, shape=[], coordinate_type="", data_type=""
    )
    response = client.get(url)
    assert response.json()["layers"]["done"] == ["layers.3", "layers.2"]
    
    # Case 4: Gap in sequence (layers.3 and layers.1 done, but NOT layers.2)
    WorkSaliencyMap.objects.all().delete()
    WorkSaliencyMap.objects.create(
        graph=graph, 
        input=input_obj, 
        coordinate="layers.3.out_0", 
        layer_name="layers.3",
        data={}, shape=[], coordinate_type="", data_type=""
    )
    WorkSaliencyMap.objects.create(
        graph=graph, 
        input=input_obj, 
        coordinate="layers.1.out_0", 
        layer_name="layers.1",
        data={}, shape=[], coordinate_type="", data_type=""
    )
    
    response = client.get(url)
    # layers.1 should not be in done list because layers.2 is missing
    assert response.json()["layers"]["done"] == ["layers.3"]
    
    # Case 5: Fill the gap
    WorkSaliencyMap.objects.create(
        graph=graph, 
        input=input_obj, 
        coordinate="layers.2.out_0", 
        layer_name="layers.2",
        data={}, shape=[], coordinate_type="", data_type=""
    )
    response = client.get(url)
    assert response.json()["layers"]["done"] == ["layers.3", "layers.2", "layers.1"]
