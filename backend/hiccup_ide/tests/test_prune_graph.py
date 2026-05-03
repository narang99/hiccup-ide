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
