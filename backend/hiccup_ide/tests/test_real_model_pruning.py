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


def _create_saliency_map_from_coord_object(
    input_obj, coordinate, layer_name, data_info
):
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
    url = (
        f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/status/"
    )
    return client.get(url)


def api_save_saliency_maps(
    client, model_alias, input_alias, workflow_name, graph_alias, payload
):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/saliency_maps/"
    return client.post(url, data=payload, content_type="application/json")


def api_finalize_pruning(client, model_alias, input_alias, workflow_name, graph_alias):
    url = f"{get_base_url(model_alias, input_alias, workflow_name, graph_alias)}/finalize_pruning/"
    return client.post(url)


def get_current_temp_data(graph, layer_names=None):
    """Get a dictionary of current data in TempPruneSaliencyMap for the given layers"""
    qs = TempPruneSaliencyMap.objects.filter(graph=graph)
    if layer_names:
        qs = qs.filter(layer_name__in=layer_names)

    return {tm.coordinate: tm.data for tm in qs}


def check_layer_data_changed(graph, layer_name, original_data_dict):
    """Returns True if any coordinate in the layer has different data from original_data_dict"""
    current_data = get_current_temp_data(graph, [layer_name])

    for coordinate, data in current_data.items():
        if coordinate in original_data_dict:
            if data != original_data_dict[coordinate]:
                return True
    return False


def assert_layer_tensor_matches(graph, layer_name, expected_contribs):
    """Asserts that the reconstructed tensor for a layer matches the expected contribution tensor"""
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


def create_threshold_payload(saliency_maps, layer_name, threshold=None):
    """Finds first coordinate in layer and creates a median threshold payload.
    Uses median of non-zero absolute values to ensure some values are actually pruned.
    """
    layer_maps = [sm for sm in saliency_maps if sm.layer_name == layer_name]
    if not layer_maps:
        raise ValueError(f"No coordinates found for layer {layer_name}")

    datas = [m.data for m in layer_maps]
    datas = torch.tensor(datas)
    threshold = threshold if threshold is not None else float(torch.median(datas))

    return {
        "items": [
            {
                "coordinate": sm.coordinate,
                "algorithm": {"type": "ThresholdAlgorithm", "threshold": threshold},
            }
            for sm in layer_maps
        ]
    }


# --- Tests ---


@pytest.mark.django_db
@requires_real_model()
@pytest.mark.parametrize(
    "layer_to_prune, upstream_affected, downstream_unaffected",
    [
        ("layers.3", ["layers.2", "layers.1", "x"], []),
        ("layers.2", ["layers.1", "x"], ["layers.3"]),
        ("x", [], ["layers.3", "layers.2", "layers.1"]),
    ],
)
def test_real_model_pruning_propagation_and_ground_truth(
    real_model_data, layer_to_prune, upstream_affected, downstream_unaffected
):
    """Consolidated test for pruning propagation, ground truth, and isolation"""
    model, input_obj, saliency_maps, _, _ = real_model_data
    client = Client()

    workflow_name = f"prune-test-{layer_to_prune}"
    graph_alias = f"graph-{layer_to_prune}"

    # 1. Initialize session
    response = api_start_pruning(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert response.status_code == 200

    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    original_temp_data = get_current_temp_data(graph)

    # 2. Apply pruning to target layer
    payload = create_threshold_payload(saliency_maps, layer_to_prune)
    target_coordinate = payload["items"][0]["coordinate"]
    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # 3. Verify target layer was modified
    target_temp_map = TempPruneSaliencyMap.objects.get(
        graph=graph, coordinate=target_coordinate
    )
    assert target_temp_map.is_modified
    assert check_layer_data_changed(graph, layer_to_prune, original_temp_data)

    # 4. Verify downstream isolation (unaffected layers)
    for layer_name in downstream_unaffected:
        assert not check_layer_data_changed(graph, layer_name, original_temp_data), (
            f"Downstream layer {layer_name} should remain unchanged"
        )
        # Verify no is_modified flags were set for downstream
        assert not TempPruneSaliencyMap.objects.filter(
            graph=graph, layer_name=layer_name, is_modified=True
        ).exists()

    # 5. Verify upstream propagation (ground truth against actual model)
    if upstream_affected:
        pruned_tensor = reconstruct_layer_tensor(graph, layer_to_prune)
        assert pruned_tensor is not None
        expected_contribs = calculate_expected_ground_truth(
            layer_to_prune, pruned_tensor
        )

        for layer_name in upstream_affected:
            # Check data actually changed from original
            assert check_layer_data_changed(graph, layer_name, original_temp_data), (
                f"Upstream layer {layer_name} should have changed due to backprop"
            )
            # Check it matches ground truth exactly
            assert_layer_tensor_matches(graph, layer_name, expected_contribs)

            # Upstream should NOT be marked as is_modified (only targets are)
            assert not TempPruneSaliencyMap.objects.filter(
                graph=graph, layer_name=layer_name, is_modified=True
            ).exists()


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

    # Apply pruning to layers.3 with a median threshold
    payload = create_threshold_payload(saliency_maps, "layers.3")
    target_coordinate = payload["items"][0]["coordinate"]
    threshold = payload["items"][0]["algorithm"]["threshold"]

    response = api_save_saliency_maps(
        client, model.alias, input_obj.alias, workflow_name, graph_alias, payload
    )
    assert response.status_code == 200

    # Check status
    status_response = api_get_status(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert status_response.json()["layers"]["done"] == ["layers.3"]
    assert status_response.json()["session_active"]

    # Finalize
    response = api_finalize_pruning(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert response.status_code == 200
    assert response.json()["committed_count"] == len(payload["items"])

    # Verify WorkSaliencyMap has the data
    work = Work.objects.get(input=input_obj, name=workflow_name)
    graph = WorkGraph.objects.get(work=work, alias=graph_alias)
    work_map = WorkSaliencyMap.objects.get(graph=graph, coordinate=target_coordinate)

    from neural_data.api.helpers import apply_algorithm

    # Get original data for verification
    original_sm = SaliencyMap.objects.get(input=input_obj, coordinate=target_coordinate)
    expected_data = apply_algorithm(
        original_sm.data, {"type": "ThresholdAlgorithm", "threshold": threshold}
    )
    assert work_map.data == expected_data

    # Verify temp maps are cleaned up
    assert TempPruneSaliencyMap.objects.filter(graph=graph).count() == 0

    # Final status should show no active session
    final_status = api_get_status(
        client, model.alias, input_obj.alias, workflow_name, graph_alias
    )
    assert not final_status.json()["session_active"]
    assert final_status.json()["layers"]["done"] == []  # Only checking temp maps now
