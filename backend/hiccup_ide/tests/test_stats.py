import pytest
from django.test import Client
from neural_data.models import Model, Input, Activation, SaliencyMap, Work, WorkGraph, WorkSaliencyMap

@pytest.fixture
def sample_data():
    model = Model.objects.create(
        alias="test-model",
        name="Test Model",
        definition={}
    )
    input_obj = Input.objects.create(
        model=model,
        alias="test-input",
        name="Test Input",
        data_path="/path/to/data"
    )
    
    # Create sample activations
    Activation.objects.create(
        input=input_obj,
        coordinate="layer1.node1",
        data=[[1.0, 2.0], [3.0, 4.0]],
        shape=[2, 2],
        layer_type="Conv2d",
        coordinate_type="output_channel"
    )
    Activation.objects.create(
        input=input_obj,
        coordinate="layer1.node2",
        data=[[-1.0, 0.0], [5.0, 10.0]],
        shape=[2, 2],
        layer_type="Conv2d",
        coordinate_type="output_channel"
    )
    
    # Create sample saliency maps
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layer1.node1",
        layer_name="layer1",
        data=[[0.1, 0.2], [0.3, 0.4]],
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="contrib"
    )
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layer1.node2",
        layer_name="layer1",
        data=[[-0.5, 0.0], [0.5, 1.5]],
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="contrib"
    )

    # Create a work and graph
    work = Work.objects.create(input=input_obj, name="test-work")
    graph = WorkGraph.objects.create(work=work, alias="test-graph")

    # Create work saliency map for node1 (overriding base)
    WorkSaliencyMap.objects.create(
        input=input_obj,
        graph=graph,
        coordinate="layer1.node1",
        layer_name="layer1",
        data=[[10.0, 20.0], [30.0, 40.0]],
        shape=[2, 2],
        coordinate_type="output_channel",
        data_type="contrib"
    )
    
    return model, input_obj, work, graph

@pytest.mark.django_db
def test_activations_stats_api(sample_data):
    model, input_obj, _, _ = sample_data
    client = Client()
    
    response = client.post(
        f"/api/models/{model.alias}/inputs/{input_obj.alias}/activations/stats/",
        data={"coordinates": ["layer1.node1", "layer1.node2"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == -1.0
    assert data["max"] == 10.0

@pytest.mark.django_db
def test_activations_stats_single_node(sample_data):
    model, input_obj, _, _ = sample_data
    client = Client()
    
    response = client.post(
        f"/api/models/{model.alias}/inputs/{input_obj.alias}/activations/stats/",
        data={"coordinates": ["layer1.node1"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == 1.0
    assert data["max"] == 4.0

@pytest.mark.django_db
def test_saliency_maps_stats_api(sample_data):
    model, input_obj, _, _ = sample_data
    client = Client()
    
    response = client.post(
        f"/api/models/{model.alias}/inputs/{input_obj.alias}/saliency_maps/stats/",
        data={"coordinates": ["layer1.node1", "layer1.node2"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == -0.5
    assert data["max"] == 1.5

@pytest.mark.django_db
def test_saliency_maps_stats_with_aliases(sample_data):
    model, input_obj, work, graph = sample_data
    client = Client()
    
    # Test with aliases - should use WorkSaliencyMap for node1 and SaliencyMap for node2
    # node1 (work): [[10.0, 20.0], [30.0, 40.0]], node2 (base): [[-0.5, 0.0], [0.5, 1.5]] -> min: -0.5, max: 40.0
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/saliency_maps/stats/?work_alias={work.name}&graph_alias={graph.alias}"
    response = client.post(
        url,
        data={"coordinates": ["layer1.node1", "layer1.node2"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == -0.5
    assert data["max"] == 40.0

@pytest.mark.django_db
def test_stats_empty_coordinates(sample_data):
    model, input_obj, _, _ = sample_data
    client = Client()
    
    response = client.post(
        f"/api/models/{model.alias}/inputs/{input_obj.alias}/activations/stats/",
        data={"coordinates": []},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == 0.0
    assert data["max"] == 0.0

@pytest.mark.django_db
def test_stats_nonexistent_coordinates(sample_data):
    model, input_obj, _, _ = sample_data
    client = Client()
    
    response = client.post(
        f"/api/models/{model.alias}/inputs/{input_obj.alias}/activations/stats/",
        data={"coordinates": ["nonexistent"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["min"] == 0.0
    assert data["max"] == 0.0
