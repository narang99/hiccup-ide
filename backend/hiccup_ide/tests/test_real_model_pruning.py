import pytest
import torch
import os
from django.test import Client
from pt_to_api.contrib_processor import process_contribs_to_coordinates
from neural_data.api.saliency import reconstruct_layer_tensor
from neural_data.models import (
    Model,
    Input,
    Work,
    WorkGraph,
    SaliencyMap,
    WorkSaliencyMap,
    TempPruneSaliencyMap,
)


# Hardcoded paths matching the saliency.py implementation
MODEL_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/model.pt"
INPUT_PATH = (
    "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/first-input-tens.pt"
)


def requires_real_model():
    """Decorator to skip tests if real model files are not available"""
    return pytest.mark.skipif(
        not (os.path.exists(MODEL_PATH) and os.path.exists(INPUT_PATH)),
        reason="Real model and input files not found",
    )


@pytest.fixture(scope="session")
def raw_real_model_data_to_persist_in_db():
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

    coord_data = process_contribs_to_coordinates(real_contribs, sample_idx=0)

    return coord_data, real_contribs, batch_inp_tens


def _create_saliency_map_from_coord_object(input_obj, coordinate, layer_name, data_info):
    return SaliencyMap.objects.create(
        input=input_obj,
        coordinate=coordinate,
        layer_name=layer_name,
        data=data_info["data"],
        shape=data_info["shape"],
        coordinate_type=data_info.get("coordinate_type", "output_channel"),
        data_type="saliency",
        output_channel=data_info.get("output_channel"),
        input_channel=data_info.get("input_channel"),
    )

@pytest.fixture
def real_model_data(raw_real_model_data_to_persist_in_db):
    """Create test data that matches the actual model architecture"""
    coord_data, real_contribs, batch_inp_tens = raw_real_model_data_to_persist_in_db
    model = Model.objects.create(
        alias="real-mnist-model",
        name="Real MNIST Model",
        definition={"nodes": [], "edges": []},
    )
    input_obj = Input.objects.create(
        model=model,
        alias="real-mnist-input",
        name="Real MNIST Input",
        data_path=INPUT_PATH,
    )

    saliency_maps = []
    for coordinate, data_info in coord_data.items():
        if coordinate.startswith("layers.3."):
            saliency_map = _create_saliency_map_from_coord_object(
                input_obj, coordinate, "layers.3", data_info
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("layers.2."):
            saliency_map = _create_saliency_map_from_coord_object(
                input_obj, coordinate, "layers.2", data_info
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("layers.1."):
            saliency_map = _create_saliency_map_from_coord_object(
                input_obj, coordinate, "layers.1", data_info
            )
            saliency_maps.append(saliency_map)
        elif coordinate.startswith("x."):
            saliency_map = _create_saliency_map_from_coord_object(
                input_obj, coordinate, "x", data_info
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


def get_base_url(model_alias, input_alias, workflow_name, graph_alias):
    return f"/api/models/{model_alias}/inputs/{input_alias}/workflows/{workflow_name}/graphs/{graph_alias}"


def api_start_pruning(client, model_alias, input_alias, workflow_name, graph_alias):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/start_pruning/"
    return client.post(url)


def api_get_status(client, model_alias, input_alias, workflow_name, graph_alias):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/status/"
    return client.get(url)


def api_save_saliency_maps(
    client, model_alias, input_alias, workflow_name, graph_alias, payload
):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/saliency_maps/"
    return client.post(url, data=payload, content_type="application/json")


def api_finalize_pruning(client, model_alias, input_alias, workflow_name, graph_alias):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/finalize_pruning/"
    return client.post(url)


@pytest.mark.django_db
@requires_real_model()
def test_real_model_layers_3_pruning_ground_truth(real_model_data):
    """Test pruning layers.3 with actual model and validate against ground truth"""
    model, input_obj, saliency_maps, _, _ = real_model_data
    client = Client()

    workflow_name = "real-model-test-3"
    graph_alias = "real-graph-3"

    # Initialize session
    response = api_start_pruning(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
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

    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    pruned_tensor = reconstruct_layer_tensor(graph, "layers.3")
    assert pruned_tensor is not None

    expected_contribs = calculate_expected_ground_truth("layers.3", pruned_tensor)

    upstream_layers = ["layers.2", "layers.1", "x"]

    for layer_name in upstream_layers:
        assert layer_name in expected_contribs

        actual_tensor = reconstruct_layer_tensor(graph, layer_name)
        expected_tensor = expected_contribs[layer_name]

        assert actual_tensor.shape == expected_tensor.shape, (
            f"Shape mismatch for {layer_name}: actual {actual_tensor.shape} vs expected {expected_tensor.shape}"
        )

        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=1e-6,
            rtol=1e-5,
            msg=f"Value mismatch for {layer_name}",
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
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

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

    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    pruned_tensor = reconstruct_layer_tensor(graph, "layers.2")
    assert pruned_tensor is not None

    # Calculate expected ground truth
    expected_contribs = calculate_expected_ground_truth("layers.2", pruned_tensor)

    # Verify upstream layers got correct propagated values
    upstream_layers = ["layers.1", "x"]

    for layer_name in upstream_layers:
        assert layer_name in expected_contribs

        actual_tensor = reconstruct_layer_tensor(graph, layer_name)
        expected_tensor = expected_contribs[layer_name]

        assert actual_tensor.shape == expected_tensor.shape, (
            f"Shape mismatch for {layer_name}: actual {actual_tensor.shape} vs expected {expected_tensor.shape}"
        )

        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=1e-6,
            rtol=1e-5,
            msg=f"Value mismatch for {layer_name}",
        )

    # Verify that layers.3 was NOT modified (downstream layer)
    layers_3_modified = TempPruneSaliencyMap.objects.filter(
        graph=graph, layer_name="layers.3", is_modified=True
    )
    assert layers_3_modified.count() == 0, (
        "Downstream layer layers.3 should not be modified"
    )


@pytest.mark.django_db
@requires_real_model()
def test_real_model_input_layer_no_propagation(real_model_data):
    """Test that pruning input layer 'x' with real model doesn't propagate (no upstream layers)"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "real-model-input-test"
    graph_alias = "real-input-graph"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Find an input layer coordinate
    x_maps = [sm for sm in saliency_maps if sm.layer_name == "x"]
    assert len(x_maps) > 0, "Need at least one x coordinate"

    target_coordinate = x_maps[0].coordinate
    original_data = x_maps[0].data

    # Store original data of all other layers
    original_temp_data = {}
    for temp_map in TempPruneSaliencyMap.objects.filter(graph=graph).exclude(
        layer_name="x"
    ):
        original_temp_data[temp_map.coordinate] = temp_map.data.copy()

    # Apply pruning to input layer
    flat_data = torch.tensor(original_data).flatten()
    threshold = float(torch.median(flat_data))

    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Verify the input layer was modified
    x_temp_map = TempPruneSaliencyMap.objects.get(
        graph=graph, coordinate=target_coordinate
    )
    assert x_temp_map.is_modified == True

    # Verify that no other layers were affected (since x has no upstream)
    for temp_map in TempPruneSaliencyMap.objects.filter(graph=graph).exclude(
        layer_name="x"
    ):
        assert not temp_map.is_modified, (
            f"Layer {temp_map.layer_name} should not be user-modified"
        )
        # Data should be unchanged (no propagation occurred)
        assert temp_map.data == original_temp_data[temp_map.coordinate], (
            f"Data for {temp_map.coordinate} should be unchanged"
        )


@pytest.mark.django_db
@requires_real_model()
def test_real_model_algorithm_correctness(real_model_data):
    """Test that algorithms are applied correctly to real model data"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "real-algorithm-test"
    graph_alias = "real-algorithm-graph"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Test layers.3 coordinates
    layers_3_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.3"][:2]
    if layers_3_maps:
        test_cases_3 = []
        for saliency_map in layers_3_maps:
            flat_data = torch.tensor(saliency_map.data).flatten()
            median_val = float(torch.median(flat_data))
            test_cases_3.append(
                {
                    "coordinate": saliency_map.coordinate,
                    "algorithm": {
                        "type": "ThresholdAlgorithm",
                        "threshold": median_val,
                    },
                    "original_data": saliency_map.data,
                }
            )

        # Calculate expected results
        from neural_data.api.helpers import apply_algorithm

        for test_case in test_cases_3:
            test_case["expected_data"] = apply_algorithm(
                test_case["original_data"], test_case["algorithm"]
            )

        payload_3 = {
            "items": [
                {"coordinate": tc["coordinate"], "algorithm": tc["algorithm"]}
                for tc in test_cases_3
            ]
        }
        response = api_save_saliency_maps(
            client, model.alias, input_obj.alias, workflow_name, graph_alias, payload_3
        )
        assert response.status_code == 200

        # Verify each algorithm was applied correctly for layers.3
        for test_case in test_cases_3:
            temp_map = TempPruneSaliencyMap.objects.get(
                graph=graph, coordinate=test_case["coordinate"]
            )
            assert temp_map.is_modified == True
            assert temp_map.data == test_case["expected_data"], (
                f"Algorithm not applied correctly to {test_case['coordinate']}"
            )


@pytest.mark.django_db
@requires_real_model()
def test_real_model_propagation_data_changes(real_model_data):
    """Test that pruning layers.3 actually changes upstream layer data (verifies backprop)"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "propagation-data-test"
    graph_alias = "data-change-test"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Store original data for upstream layers before any pruning
    upstream_layers = ["layers.2", "layers.1", "x"]
    original_data = {}

    for layer_name in upstream_layers:
        temp_maps = TempPruneSaliencyMap.objects.filter(
            graph=graph, layer_name=layer_name
        )[:2]  # Take first 2 coordinates
        for tm in temp_maps:
            original_data[tm.coordinate] = (
                tm.data.copy() if isinstance(tm.data, list) else tm.data
            )

    # Find a layers.3 coordinate to prune
    layers_3_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.3"]
    assert len(layers_3_maps) > 0, "Need at least one layers.3 coordinate"
    target_coordinate = layers_3_maps[0].coordinate

    # Apply pruning to layers.3 which should trigger backpropagation
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.5},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Verify that upstream layers have different data after backpropagation
    changes_detected = 0
    for coordinate, orig_data in original_data.items():
        updated_tm = TempPruneSaliencyMap.objects.get(
            graph=graph, coordinate=coordinate
        )

        # Check if data actually changed (this is the key test for backprop)
        if updated_tm.data != orig_data:
            changes_detected += 1
            print(f"✓ Backprop detected: {coordinate} data changed")
        else:
            print(f"✗ No change: {coordinate} data unchanged")

    assert changes_detected > 0, (
        "Expected at least some upstream coordinates to change due to backpropagation"
    )

    # Verify the pruned layer itself was modified
    pruned_layer = TempPruneSaliencyMap.objects.get(
        graph=graph, coordinate=target_coordinate
    )
    assert pruned_layer.is_modified is True


@pytest.mark.django_db
@requires_real_model()
def test_real_model_layer_isolation_downstream_unchanged(real_model_data):
    """Test that pruning layers.2 does NOT affect downstream layers.3 (verifies isolation)"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "isolation-test"
    graph_alias = "downstream-isolation"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Store original data for downstream layers.3
    layers_3_temp_maps = TempPruneSaliencyMap.objects.filter(
        graph=graph, layer_name="layers.3"
    )[:3]
    original_layers_3_data = {}
    for tm in layers_3_temp_maps:
        original_layers_3_data[tm.coordinate] = (
            tm.data.copy() if isinstance(tm.data, list) else tm.data
        )

    # Find a layers.2 coordinate to prune
    layers_2_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.2"]
    assert len(layers_2_maps) > 0, "Need at least one layers.2 coordinate"
    target_coordinate = layers_2_maps[0].coordinate

    # Apply pruning to layers.2 (middle layer)
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.5},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Verify that downstream layers.3 data did NOT change
    unchanged_count = 0
    for coordinate, orig_data in original_layers_3_data.items():
        updated_tm = TempPruneSaliencyMap.objects.get(
            graph=graph, coordinate=coordinate
        )

        if updated_tm.data == orig_data:
            unchanged_count += 1
            print(f"✓ Downstream isolation verified: {coordinate} data unchanged")
        else:
            print(f"✗ Unexpected change: {coordinate} data changed")

    assert unchanged_count == len(original_layers_3_data), (
        "All downstream layers.3 data should remain unchanged"
    )

    # Verify layers.2 itself was modified
    pruned_layer = TempPruneSaliencyMap.objects.get(
        graph=graph, coordinate=target_coordinate
    )
    assert pruned_layer.is_modified is True

    # Verify upstream layers (layers.1, x) should have changes due to backprop
    upstream_changes = 0
    upstream_layers = ["layers.1", "x"]
    for layer_name in upstream_layers:
        temp_maps = TempPruneSaliencyMap.objects.filter(
            graph=graph, layer_name=layer_name
        )[:2]
        for tm in temp_maps:
            # Compare with original SaliencyMap data
            orig_sm = SaliencyMap.objects.get(input=input_obj, coordinate=tm.coordinate)
            if tm.data != orig_sm.data:
                upstream_changes += 1
                break

    # We expect at least some upstream changes from backprop
    assert upstream_changes > 0, (
        "Expected upstream layers to change due to backpropagation"
    )


@pytest.mark.django_db
@requires_real_model()
def test_real_model_input_layer_no_upstream_changes(real_model_data):
    """Test that pruning input layer 'x' does not change any other layers (no upstream to backprop to)"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "input-layer-test"
    graph_alias = "input-isolation"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Store original data for ALL other layers (not x)
    other_layers = ["layers.3", "layers.2", "layers.1"]
    original_data = {}

    for layer_name in other_layers:
        temp_maps = TempPruneSaliencyMap.objects.filter(
            graph=graph, layer_name=layer_name
        )[:2]
        for tm in temp_maps:
            original_data[tm.coordinate] = (
                tm.data.copy() if isinstance(tm.data, list) else tm.data
            )

    # Find an x layer coordinate to prune
    x_maps = [sm for sm in saliency_maps if sm.layer_name == "x"]
    assert len(x_maps) > 0, "Need at least one x coordinate"
    target_coordinate = x_maps[0].coordinate

    # Apply pruning to input layer 'x'
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.5},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Verify that NO other layers changed (since x has no upstream)
    unchanged_count = 0
    for coordinate, orig_data in original_data.items():
        updated_tm = TempPruneSaliencyMap.objects.get(
            graph=graph, coordinate=coordinate
        )

        if updated_tm.data == orig_data:
            unchanged_count += 1
            print(f"✓ No upstream propagation: {coordinate} data unchanged")
        else:
            print(f"✗ Unexpected change: {coordinate} data changed")

    assert unchanged_count == len(original_data), (
        "No other layers should change when pruning input layer"
    )

    # Verify only the input layer itself was modified
    modified_temp_maps = TempPruneSaliencyMap.objects.filter(
        graph=graph, is_modified=True
    )
    assert modified_temp_maps.count() == 1, (
        "Only the pruned input coordinate should be modified"
    )
    assert modified_temp_maps.first().coordinate == target_coordinate


@pytest.mark.django_db
@requires_real_model()
def test_real_model_algorithm_uses_original_data(real_model_data):
    """Test that pruning algorithms are applied to original SaliencyMap data, not propagated data"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "algorithm-original-test"
    graph_alias = "original-data-test"

    # Initialize session
    api_start_pruning(client, model.alias, input_obj.alias, workflow_name, graph_alias)

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)

    # Find a layers.3 coordinate to prune
    layers_3_maps = [sm for sm in saliency_maps if sm.layer_name == "layers.3"]
    target_coordinate = layers_3_maps[0].coordinate

    # Get the original SaliencyMap data
    original_saliency_map = SaliencyMap.objects.get(
        input=input_obj, coordinate=target_coordinate
    )
    original_data = original_saliency_map.data

    # Apply algorithm to the coordinate
    payload = {
        "items": [
            {
                "coordinate": target_coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.5},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Get the result in TempPruneSaliencyMap
    pruned_temp_map = TempPruneSaliencyMap.objects.get(
        graph=graph, coordinate=target_coordinate
    )
    assert pruned_temp_map.is_modified is True

    # Manually apply the algorithm to the original data to verify it was used as source
    from neural_data.api.helpers import apply_algorithm

    expected_data = apply_algorithm(
        original_data, {"type": "ThresholdAlgorithm", "threshold": 0.5}
    )

    # The temp map should match the manually applied algorithm result
    assert pruned_temp_map.data == expected_data, (
        "Algorithm should be applied to original SaliencyMap data"
    )
    print(f"✓ Algorithm correctly applied to original data for {target_coordinate}")


@pytest.mark.django_db
@requires_real_model()
def test_real_model_full_workflow_with_finalization(real_model_data):
    """Test complete real model workflow including finalization"""
    model, input_obj, saliency_maps, original_contribs, batch_inp_tens = real_model_data
    client = Client()

    workflow_name = "real-full-workflow"
    graph_alias = "real-full-graph"

    # Initialize session
    response = api_start_pruning(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert response.status_code == 200

    # Apply pruning to layers.3
    layers_3_map = next(sm for sm in saliency_maps if sm.layer_name == "layers.3")
    flat_data = torch.tensor(layers_3_map.data).flatten()
    threshold = float(torch.median(flat_data))

    payload = {
        "items": [
            {
                "coordinate": layers_3_map.coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold},
            }
        ]
    }
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Check status
    status_response = api_get_status(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert status_response.json()["layers"]["done"] == ["layers.3"]
    assert status_response.json()["session_active"] == True

    # Finalize
    response = api_finalize_pruning(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert response.status_code == 200
    assert response.json()["committed_count"] == 1

    # Verify WorkSaliencyMap has the data
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    work_map = WorkSaliencyMap.objects.get(
        graph=graph, coordinate=layers_3_map.coordinate
    )

    from neural_data.api.helpers import apply_algorithm

    expected_data = apply_algorithm(
        layers_3_map.data, {"type": "ThresholdAlgorithm", "threshold": threshold}
    )
    assert work_map.data == expected_data

    # Verify temp maps are cleaned up
    assert TempPruneSaliencyMap.objects.filter(graph=graph).count() == 0

    # Final status should show no active session
    final_status = api_get_status(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert final_status.json()["session_active"] == False
    assert final_status.json()["layers"]["done"] == []  # Only checking temp maps now
