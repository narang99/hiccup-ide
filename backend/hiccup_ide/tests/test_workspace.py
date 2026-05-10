import pytest
from django.test import Client
from neural_data.models import Model, Input, Work, WorkGraph, WorkSaliencyMap

@pytest.mark.django_db
def test_get_workspace_with_graph_indicator():
    # Setup data
    model = Model.objects.create(
        alias="test-model",
        name="Test Model",
        definition={"nodes": [], "edges": []}
    )
    input_obj = Input.objects.create(
        alias="test-input",
        model=model,
        name="Test Input",
        data_path="test/path",
        category="test"
    )
    
    # Work 1: Has graph, but NO WorkSaliencyMap
    work_no_saliency = Work.objects.create(input=input_obj, name="Graph No Saliency")
    graph_no_saliency = WorkGraph.objects.create(work=work_no_saliency)
    
    # Work 2: Has graph AND WorkSaliencyMap
    work_with_saliency = Work.objects.create(input=input_obj, name="Graph With Saliency")
    graph_with_saliency = WorkGraph.objects.create(work=work_with_saliency)
    WorkSaliencyMap.objects.create(
        input=input_obj, graph=graph_with_saliency, coordinate="1,1", 
        layer_name="layer1", data={}, shape=[], coordinate_type="test", data_type="test"
    )
    
    # Work 3: No graph at all
    work_no_graph = Work.objects.create(input=input_obj, name="No Graph")
    
    client = Client()
    response = client.get("/api/workspace/")
    
    assert response.status_code == 200
    data = response.json()
    
    works_data = data[0]["inputs"][0]["works"]
    
    # Graph No Saliency -> has_graph should be False
    assert works_data[0]["name"] == "Graph No Saliency"
    assert works_data[0]["has_graph"] is False
    
    # Graph With Saliency -> has_graph should be True
    assert works_data[1]["name"] == "Graph With Saliency"
    assert works_data[1]["has_graph"] is True
    
    # No Graph -> has_graph should be False
    assert works_data[2]["name"] == "No Graph"
    assert works_data[2]["has_graph"] is False
