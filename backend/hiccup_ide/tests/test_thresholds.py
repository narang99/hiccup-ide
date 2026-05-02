import pytest
from django.test import Client

from neural_data.models import Model, Input, Work, LayerThreshold


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return Model.objects.create(
        alias="test-model",
        name="Test Neural Network",
        definition={"nodes": [], "edges": []}
    )


@pytest.fixture
def sample_input(sample_model):
    """Create a sample input for testing."""
    return Input.objects.create(
        model=sample_model,
        alias="test-input",
        name="Test Input",
        data_path="/path/to/test/data"
    )


@pytest.fixture
def sample_work(sample_input):
    """Create a sample work/workflow for testing."""
    return Work.objects.create(
        input=sample_input,
        name="test-workflow"
    )


@pytest.fixture
def threshold_algorithm():
    """Sample ThresholdAlgorithm data."""
    return {
        "type": "ThresholdAlgorithm",
        "threshold": 0.5
    }


@pytest.fixture
def id_algorithm():
    """Sample Id algorithm data."""
    return {
        "type": "Id"
    }


# LayerThreshold Model Tests

@pytest.mark.django_db
def test_create_layer_threshold_with_threshold_algorithm(sample_work, sample_model, threshold_algorithm):
    """Test creating a LayerThreshold with ThresholdAlgorithm."""
    threshold = LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.0",
        slider_value=75.0,
        algorithm=threshold_algorithm
    )
    
    assert threshold.pk is not None
    assert threshold.layer_id == "layers.0"
    assert threshold.slider_value == 75.0
    assert threshold.algorithm == threshold_algorithm
    assert threshold.algorithm["type"] == "ThresholdAlgorithm"
    assert threshold.algorithm["threshold"] == 0.5


@pytest.mark.django_db
def test_create_layer_threshold_with_id_algorithm(sample_work, sample_model, id_algorithm):
    """Test creating a LayerThreshold with Id algorithm."""
    threshold = LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.1",
        slider_value=90.0,
        algorithm=id_algorithm
    )
    
    assert threshold.layer_id == "layers.1"
    assert threshold.slider_value == 90.0
    assert threshold.algorithm == id_algorithm
    assert threshold.algorithm["type"] == "Id"


@pytest.mark.django_db
def test_create_layer_threshold_with_empty_algorithm(sample_work, sample_model):
    """Test creating a LayerThreshold with default empty algorithm."""
    threshold = LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.2",
        slider_value=50.0,
        # algorithm defaults to empty dict
    )
    
    assert threshold.slider_value == 50.0
    assert threshold.algorithm == {}


@pytest.mark.django_db
def test_layer_threshold_unique_constraint(sample_work, sample_model, threshold_algorithm):
    """Test that work+model+layer_id must be unique."""
    # Create first threshold
    LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.0",
        algorithm=threshold_algorithm
    )
    
    # Attempt to create duplicate should raise exception
    with pytest.raises(Exception):  # IntegrityError
        LayerThreshold.objects.create(
            work=sample_work,
            model=sample_model,
            layer_id="layers.0",  # Same combination
            algorithm=threshold_algorithm
        )


@pytest.mark.django_db
def test_different_works_same_layer(sample_input, sample_model, threshold_algorithm):
    """Test that different works can have thresholds for the same layer."""
    work1 = Work.objects.create(input=sample_input, name="workflow1")
    work2 = Work.objects.create(input=sample_input, name="workflow2")
    
    # Create thresholds with same layer in different works
    threshold1 = LayerThreshold.objects.create(
        work=work1,
        model=sample_model,
        layer_id="layers.0",
        algorithm=threshold_algorithm
    )
    
    threshold2 = LayerThreshold.objects.create(
        work=work2,
        model=sample_model,
        layer_id="layers.0",  # Same layer, different work
        algorithm={"type": "Id"}
    )
    
    assert threshold1.pk != threshold2.pk
    assert threshold1.work != threshold2.work
    assert threshold1.layer_id == threshold2.layer_id


# LayerThreshold API Tests

@pytest.mark.django_db
def test_save_threshold_algorithm_via_api(sample_model, sample_input, threshold_algorithm):
    """Test saving a threshold algorithm via API."""
    client = Client()
    
    response = client.post(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/",
        data={
            "layer_id": "layers.0",
            "slider_value": 75.0,
            "algorithm": threshold_algorithm
        },
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["layer_id"] == "layers.0"
    assert data["slider_value"] == 75.0
    assert data["algorithm"] == threshold_algorithm
    assert data["algorithm"]["type"] == "ThresholdAlgorithm"
    assert data["algorithm"]["threshold"] == 0.5


@pytest.mark.django_db
def test_save_id_algorithm_via_api(sample_model, sample_input, id_algorithm):
    """Test saving an Id algorithm via API."""
    client = Client()
    
    response = client.post(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/",
        data={
            "layer_id": "layers.1",
            "slider_value": 90.0,
            "algorithm": id_algorithm
        },
        content_type="application/json"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["layer_id"] == "layers.1"
    assert data["slider_value"] == 90.0
    assert data["algorithm"] == id_algorithm
    assert data["algorithm"]["type"] == "Id"


@pytest.mark.django_db
def test_update_existing_threshold_via_api(sample_model, sample_input, threshold_algorithm, id_algorithm):
    """Test updating an existing threshold algorithm via API."""
    client = Client()
    
    # First save
    response1 = client.post(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/",
        data={
            "layer_id": "layers.0",
            "slider_value": 60.0,
            "algorithm": threshold_algorithm
        },
        content_type="application/json"
    )
    assert response1.status_code == 200
    original_id = response1.json()["id"]
    
    # Update with different algorithm and slider value
    response2 = client.post(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/",
        data={
            "layer_id": "layers.0",  # Same layer_id
            "slider_value": 80.0,   # New slider value
            "algorithm": id_algorithm
        },
        content_type="application/json"
    )
    
    assert response2.status_code == 200
    data = response2.json()
    assert data["id"] == original_id  # Same record updated
    assert data["slider_value"] == 80.0  # New slider value
    assert data["algorithm"] == id_algorithm  # New algorithm


@pytest.mark.django_db
def test_get_workflow_thresholds_via_api(sample_model, sample_input):
    """Test retrieving workflow thresholds via API."""
    client = Client()
    
    # Save multiple thresholds
    algorithms = [
        {"layer_id": "layers.0", "slider_value": 30.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.3}},
        {"layer_id": "layers.1", "slider_value": 100.0, "algorithm": {"type": "Id"}},
        {"layer_id": "layers.2", "slider_value": 80.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.8}}
    ]
    
    for alg_data in algorithms:
        client.post(
            f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/",
            data=alg_data,
            content_type="application/json"
        )
    
    # Retrieve all thresholds
    response = client.get(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/test-workflow/thresholds/"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    
    # Verify algorithms and slider values are preserved
    layer_0 = next(item for item in data if item["layer_id"] == "layers.0")
    layer_1 = next(item for item in data if item["layer_id"] == "layers.1")
    layer_2 = next(item for item in data if item["layer_id"] == "layers.2")
    
    assert layer_0["slider_value"] == 30.0
    assert layer_0["algorithm"]["type"] == "ThresholdAlgorithm"
    assert layer_0["algorithm"]["threshold"] == 0.3
    assert layer_1["slider_value"] == 100.0
    assert layer_1["algorithm"]["type"] == "Id"
    assert layer_2["slider_value"] == 80.0
    assert layer_2["algorithm"]["type"] == "ThresholdAlgorithm"
    assert layer_2["algorithm"]["threshold"] == 0.8


@pytest.mark.django_db
def test_get_empty_workflow_thresholds(sample_model, sample_input):
    """Test retrieving thresholds for a workflow with no saved thresholds."""
    client = Client()
    
    response = client.get(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/empty-workflow/thresholds/"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


@pytest.mark.django_db
def test_threshold_api_model_not_found():
    """Test 404 when model doesn't exist."""
    client = Client()
    
    response = client.post(
        "/api/models/non-existent/inputs/test/workflows/test/thresholds/",
        data={"layer_id": "test", "slider_value": 50.0, "algorithm": {"type": "Id"}},
        content_type="application/json"
    )
    
    assert response.status_code == 404


@pytest.mark.django_db
def test_threshold_api_input_not_found(sample_model):
    """Test 404 when input doesn't exist."""
    client = Client()
    
    response = client.post(
        f"/api/models/{sample_model.alias}/inputs/non-existent/workflows/test/thresholds/",
        data={"layer_id": "test", "slider_value": 50.0, "algorithm": {"type": "Id"}},
        content_type="application/json"
    )
    
    assert response.status_code == 404


@pytest.mark.django_db
def test_complex_algorithm_structures(sample_model, sample_input):
    """Test saving and retrieving complex algorithm structures."""
    client = Client()
    
    complex_algorithms = [
        {
            "layer_id": "complex.layer.1",
            "slider_value": 12.3,
            "algorithm": {
                "type": "ThresholdAlgorithm",
                "threshold": 0.123456789,
                "metadata": {
                    "computed_at": "2024-01-01T00:00:00Z",
                    "source": "user_interaction"
                }
            }
        },
        {
            "layer_id": "complex.layer.2",
            "slider_value": 100.0,
            "algorithm": {
                "type": "Id",
                "config": {
                    "preserve_original": True,
                    "debug_mode": False
                }
            }
        }
    ]
    
    # Save complex algorithms
    for alg_data in complex_algorithms:
        response = client.post(
            f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/complex-workflow/thresholds/",
            data=alg_data,
            content_type="application/json"
        )
        assert response.status_code == 200
        # Verify the complex structure is preserved
        assert response.json()["algorithm"] == alg_data["algorithm"]
    
    # Retrieve and verify complex algorithms
    response = client.get(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/complex-workflow/thresholds/"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Verify complex nested structures are preserved
    complex_1 = next(item for item in data if item["layer_id"] == "complex.layer.1")
    complex_2 = next(item for item in data if item["layer_id"] == "complex.layer.2")
    
    assert complex_1["algorithm"]["metadata"]["computed_at"] == "2024-01-01T00:00:00Z"
    assert complex_2["algorithm"]["config"]["preserve_original"] is True


# LayerThreshold Model Relationships

@pytest.mark.django_db
def test_threshold_work_relationship(sample_work, sample_model, threshold_algorithm):
    """Test that thresholds are properly related to works."""
    threshold = LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.0",
        algorithm=threshold_algorithm
    )
    
    # Test forward relationship
    assert threshold.work == sample_work
    
    # Test reverse relationship
    work_thresholds = sample_work.thresholds.all()
    assert threshold in work_thresholds
    assert work_thresholds.count() == 1


@pytest.mark.django_db
def test_threshold_model_relationship(sample_work, sample_model, threshold_algorithm):
    """Test that thresholds are properly related to models."""
    threshold = LayerThreshold.objects.create(
        work=sample_work,
        model=sample_model,
        layer_id="layers.0",
        algorithm=threshold_algorithm
    )
    
    # Test forward relationship
    assert threshold.model == sample_model
    
    # Test reverse relationship
    model_thresholds = sample_model.thresholds.all()
    assert threshold in model_thresholds
    assert model_thresholds.count() == 1


@pytest.mark.django_db
def test_bulk_threshold_operations(sample_work, sample_model):
    """Test creating and querying multiple thresholds efficiently."""
    
    # Create multiple thresholds
    thresholds_data = []
    for layer_idx in range(5):
        thresholds_data.append(LayerThreshold(
            work=sample_work,
            model=sample_model,
            layer_id=f"layers.{layer_idx}",
            algorithm={
                "type": "ThresholdAlgorithm",
                "threshold": 0.1 * layer_idx
            }
        ))
    
    # Bulk create
    LayerThreshold.objects.bulk_create(thresholds_data)
    
    # Verify counts
    total_thresholds = LayerThreshold.objects.filter(work=sample_work).count()
    assert total_thresholds == 5
    
    # Test filtering by algorithm type
    threshold_algorithms = LayerThreshold.objects.filter(
        work=sample_work,
        algorithm__type="ThresholdAlgorithm"
    ).count()
    assert threshold_algorithms == 5
    
    # Test filtering by layer ID instead (more reliable than floating point comparison)
    specific_threshold = LayerThreshold.objects.filter(
        work=sample_work,
        layer_id="layers.3"
    ).first()
    assert specific_threshold is not None
    assert abs(specific_threshold.algorithm["threshold"] - 0.3) < 0.00001  # Handle floating point precision


@pytest.mark.django_db
def test_algorithm_field_default_behavior():
    """Test that algorithm field defaults to empty dict for existing functionality."""
    model = Model.objects.create(alias="test", name="Test", definition={})
    input_obj = Input.objects.create(model=model, alias="test", name="Test", data_path="/test")
    work = Work.objects.create(input=input_obj, name="test")
    
    # Create without specifying algorithm
    threshold = LayerThreshold.objects.create(
        work=work,
        model=model,
        layer_id="test.layer"
        # algorithm defaults to empty dict
    )
    
    assert threshold.algorithm == {}
    assert isinstance(threshold.algorithm, dict)


@pytest.mark.django_db
def test_migration_compatibility():
    """Test that the algorithm field works correctly after migration from slider_value."""
    model = Model.objects.create(alias="test", name="Test", definition={})
    input_obj = Input.objects.create(model=model, alias="test", name="Test", data_path="/test")
    work = Work.objects.create(input=input_obj, name="test")
    
    # Create threshold with empty algorithm (simulates migrated data)
    threshold = LayerThreshold.objects.create(
        work=work,
        model=model,
        layer_id="migrated.layer",
        slider_value=100.0,  # Default slider value
        algorithm={}  # Empty like migrated data
    )
    
    # Should be able to update with real algorithm
    threshold.slider_value = 65.0
    threshold.algorithm = {"type": "ThresholdAlgorithm", "threshold": 0.75}
    threshold.save()
    
    # Verify update worked
    updated_threshold = LayerThreshold.objects.get(id=threshold.pk)
    assert updated_threshold.slider_value == 65.0
    assert updated_threshold.algorithm["type"] == "ThresholdAlgorithm"
    assert updated_threshold.algorithm["threshold"] == 0.75


@pytest.mark.django_db 
def test_slider_value_and_algorithm_relationship(sample_model, sample_input):
    """Test that slider_value and algorithm can be stored and retrieved together."""
    client = Client()
    
    # Test various slider values with corresponding algorithms
    test_cases = [
        {"slider": 0.0, "algorithm": {"type": "Id"}},
        {"slider": 25.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.25}},
        {"slider": 50.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.5}},
        {"slider": 75.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 0.75}},
        {"slider": 100.0, "algorithm": {"type": "ThresholdAlgorithm", "threshold": 1.0}}
    ]
    
    # Save all test cases
    for i, case in enumerate(test_cases):
        response = client.post(
            f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/slider-test/thresholds/",
            data={
                "layer_id": f"layer.{i}",
                "slider_value": case["slider"],
                "algorithm": case["algorithm"]
            },
            content_type="application/json"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["slider_value"] == case["slider"]
        assert data["algorithm"] == case["algorithm"]
    
    # Retrieve all and verify relationship is preserved
    response = client.get(
        f"/api/models/{sample_model.alias}/inputs/{sample_input.alias}/workflows/slider-test/thresholds/"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    
    # Verify each case maintains its slider-algorithm relationship
    for item in data:
        layer_idx = int(item["layer_id"].split(".")[1])
        expected_case = test_cases[layer_idx]
        assert item["slider_value"] == expected_case["slider"]
        assert item["algorithm"] == expected_case["algorithm"]