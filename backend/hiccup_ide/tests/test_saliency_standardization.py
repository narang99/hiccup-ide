import pytest
from django.test import Client
from neural_data.models import Model, Input, SaliencyMap

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
    SaliencyMap.objects.create(
        input=input_obj,
        coordinate="layer2.node1",
        layer_name="layer2",
        data=[[1.0]],
        shape=[1, 1],
        coordinate_type="output_channel",
        data_type="contrib"
    )
    
    return model, input_obj

@pytest.mark.django_db
def test_get_layer_saliency_maps(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/saliency_maps/layers/layer1/"
    response = client.get(url)
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["coordinate"] == "layer1.node1"
    assert data["items"][0]["layer_name"] == "layer1"
    assert data["items"][1]["coordinate"] == "layer1.node2"
    assert data["items"][1]["layer_name"] == "layer1"

@pytest.mark.django_db
def test_get_batch_saliency_maps(sample_data):
    model, input_obj = sample_data
    client = Client()
    
    url = f"/api/models/{model.alias}/inputs/{input_obj.alias}/saliency_maps/batch/"
    response = client.post(
        url,
        data={"coordinates": ["layer1.node1", "layer2.node1"]},
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    
    coords = [item["coordinate"] for item in data["items"]]
    assert "layer1.node1" in coords
    assert "layer2.node1" in coords
    
    layer_names = [item["layer_name"] for item in data["items"]]
    assert "layer1" in layer_names
    assert "layer2" in layer_names
