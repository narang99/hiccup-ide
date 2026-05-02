import pytest
from django.test import Client

from neural_data.models import Model, Weight


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return Model.objects.create(
        alias="test-model",
        name="Test Neural Network",
        definition={"nodes": [], "edges": []}
    )


@pytest.fixture
def sample_weight_data():
    """Sample weight data in the format expected by the Weight model."""
    return {
        "kernel_weight": {
            "data": [
                [-0.12345, 0.67890, -0.24681],
                [0.13579, -0.86420, 0.97531],
                [0.46802, -0.35791, 0.15926]
            ],
            "shape": [3, 3],
            "layer_type": "Conv2d",
            "coordinate_type": "input_output_channel",
            "data_type": "weights"
        },
        "bias_weight": {
            "data": 0.123456789,
            "shape": [],
            "layer_type": "Conv2d",
            "coordinate_type": "output_channel_bias",
            "data_type": "bias"
        }
    }


# Weight Model Tests

@pytest.mark.django_db
def test_create_kernel_weight(sample_model, sample_weight_data):
    """Test creating a kernel weight entry."""
    weight_data = sample_weight_data["kernel_weight"]
    
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.in_0",
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    assert weight.id is not None
    assert weight.coordinate == "layers.0.out_0.in_0"
    assert weight.data == weight_data["data"]
    assert weight.shape == [3, 3]
    assert weight.layer_type == "Conv2d"
    assert weight.coordinate_type == "input_output_channel"
    assert weight.data_type == "weights"
    assert str(weight) == "test-model - layers.0.out_0.in_0 (weights)"


@pytest.mark.django_db
def test_create_bias_weight(sample_model, sample_weight_data):
    """Test creating a bias weight entry."""
    bias_data = sample_weight_data["bias_weight"]
    
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.bias",
        data=bias_data["data"],
        shape=bias_data["shape"],
        layer_type=bias_data["layer_type"],
        coordinate_type=bias_data["coordinate_type"],
        data_type=bias_data["data_type"]
    )
    
    assert weight.coordinate == "layers.0.out_0.bias"
    assert weight.data == 0.123456789
    assert weight.shape == []
    assert weight.coordinate_type == "output_channel_bias"
    assert weight.data_type == "bias"
    assert str(weight) == "test-model - layers.0.out_0.bias (bias)"


@pytest.mark.django_db
def test_weight_unique_constraint(sample_model, sample_weight_data):
    """Test that model+coordinate must be unique."""
    weight_data = sample_weight_data["kernel_weight"]
    
    # Create first weight
    Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.in_0",
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    # Attempt to create duplicate should raise exception
    with pytest.raises(Exception):  # IntegrityError
        Weight.objects.create(
            model=sample_model,
            coordinate="layers.0.out_0.in_0",  # Same coordinate
            data=weight_data["data"],
            shape=weight_data["shape"],
            layer_type=weight_data["layer_type"],
            coordinate_type=weight_data["coordinate_type"],
            data_type=weight_data["data_type"]
        )


@pytest.mark.django_db
def test_different_models_same_coordinate(sample_weight_data):
    """Test that different models can have weights with the same coordinate."""
    weight_data = sample_weight_data["kernel_weight"]
    
    model1 = Model.objects.create(alias="model1", name="Model 1", definition={})
    model2 = Model.objects.create(alias="model2", name="Model 2", definition={})
    
    # Create weights with same coordinate in different models
    weight1 = Weight.objects.create(
        model=model1,
        coordinate="layers.0.out_0.in_0",
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    weight2 = Weight.objects.create(
        model=model2,
        coordinate="layers.0.out_0.in_0",  # Same coordinate, different model
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    assert weight1.id != weight2.id
    assert weight1.model != weight2.model
    assert weight1.coordinate == weight2.coordinate


# Weight API Tests

@pytest.mark.django_db
def test_get_kernel_weight_via_api(sample_model, sample_weight_data):
    """Test retrieving a kernel weight via API."""
    weight_data = sample_weight_data["kernel_weight"]
    client = Client()
    
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.in_0",
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    response = client.get(f"/api/models/{sample_model.alias}/weights/single/{weight.coordinate}/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == weight.id
    assert data["coordinate"] == "layers.0.out_0.in_0"
    assert data["data"] == weight_data["data"]
    assert data["shape"] == [3, 3]
    assert data["layer_type"] == "Conv2d"
    assert data["coordinate_type"] == "input_output_channel"
    assert data["data_type"] == "weights"


@pytest.mark.django_db
def test_get_bias_weight_via_api(sample_model, sample_weight_data):
    """Test retrieving a bias weight via API."""
    bias_data = sample_weight_data["bias_weight"]
    client = Client()
    
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.bias",
        data=bias_data["data"],
        shape=bias_data["shape"],
        layer_type=bias_data["layer_type"],
        coordinate_type=bias_data["coordinate_type"],
        data_type=bias_data["data_type"]
    )
    
    response = client.get(f"/api/models/{sample_model.alias}/weights/single/{weight.coordinate}/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["coordinate"] == "layers.0.out_0.bias"
    assert data["data"] == 0.123456789
    assert data["shape"] == []
    assert data["coordinate_type"] == "output_channel_bias"
    assert data["data_type"] == "bias"


@pytest.mark.django_db
def test_get_weight_not_found(sample_model):
    """Test 404 when weight doesn't exist."""
    client = Client()
    
    response = client.get(f"/api/models/{sample_model.alias}/weights/non.existent.coordinate/")
    
    assert response.status_code == 404


@pytest.mark.django_db
def test_get_weight_model_not_found():
    """Test 404 when model doesn't exist."""
    client = Client()
    
    response = client.get("/api/models/non-existent-model/weights/some.coordinate/")
    
    assert response.status_code == 404


@pytest.mark.django_db
def test_coordinates_with_special_characters(sample_model, sample_weight_data):
    """Test coordinates with dots and underscores work correctly."""
    weight_data = sample_weight_data["kernel_weight"]
    client = Client()
    
    coordinates = [
        "layers.0.out_0.in_0",
        "layers.2.out_15.in_7",
        "conv_block.0.out_3.bias",
        "feature_extractor.layers.5.out_10.in_2"
    ]
    
    for coord in coordinates:
        weight = Weight.objects.create(
            model=sample_model,
            coordinate=coord,
            data=weight_data["data"],
            shape=weight_data["shape"],
            layer_type=weight_data["layer_type"],
            coordinate_type=weight_data["coordinate_type"],
            data_type=weight_data["data_type"]
        )
        
        response = client.get(f"/api/models/{sample_model.alias}/weights/single/{coord}/")
        assert response.status_code == 200
        assert response.json()["coordinate"] == coord


# Weight Model Relationships

@pytest.mark.django_db
def test_weight_model_relationship(sample_model, sample_weight_data):
    """Test that weights are properly related to models."""
    weight_data = sample_weight_data["kernel_weight"]
    
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.in_0",
        data=weight_data["data"],
        shape=weight_data["shape"],
        layer_type=weight_data["layer_type"],
        coordinate_type=weight_data["coordinate_type"],
        data_type=weight_data["data_type"]
    )
    
    # Test forward relationship
    assert weight.model == sample_model
    
    # Test reverse relationship
    model_weights = sample_model.weights.all()
    assert weight in model_weights
    assert model_weights.count() == 1


@pytest.mark.django_db
def test_bulk_weight_operations(sample_model):
    """Test creating and querying multiple weights efficiently."""
    
    # Create multiple weights
    weights_data = []
    for layer in range(2):
        for out_ch in range(3):
            for in_ch in range(2):
                weights_data.append(Weight(
                    model=sample_model,
                    coordinate=f"layers.{layer}.out_{out_ch}.in_{in_ch}",
                    data=[[0.1, 0.2], [0.3, 0.4]],
                    shape=[2, 2],
                    layer_type="Conv2d",
                    coordinate_type="input_output_channel",
                    data_type="weights"
                ))
            
            # Add bias for each output channel
            weights_data.append(Weight(
                model=sample_model,
                coordinate=f"layers.{layer}.out_{out_ch}.bias",
                data=0.1,
                shape=[],
                layer_type="Conv2d",
                coordinate_type="output_channel_bias",
                data_type="bias"
            ))
    
    # Bulk create
    Weight.objects.bulk_create(weights_data)
    
    # Verify counts
    total_weights = Weight.objects.filter(model=sample_model).count()
    assert total_weights == 18  # (3 out * 2 in + 3 bias) * 2 layers = 18
    
    # Test filtering by data type
    kernel_weights = Weight.objects.filter(model=sample_model, data_type="weights").count()
    bias_weights = Weight.objects.filter(model=sample_model, data_type="bias").count()
    
    assert kernel_weights == 12  # 3 out * 2 in * 2 layers
    assert bias_weights == 6    # 3 out * 2 layers
    
    # Test filtering by layer
    layer_0_weights = Weight.objects.filter(
        model=sample_model, 
        coordinate__startswith="layers.0"
    ).count()
    assert layer_0_weights == 9  # (3 out * 2 in + 3 bias) = 9


@pytest.mark.django_db
def test_weight_update_on_reload(sample_model, sample_weight_data):
    """Test that existing weights are updated when reloading data."""
    weight_data = sample_weight_data["kernel_weight"]
    
    # Create initial weight
    weight = Weight.objects.create(
        model=sample_model,
        coordinate="layers.0.out_0.in_0",
        data=[[1, 2], [3, 4]],  # Original data
        shape=[2, 2],
        layer_type="Conv2d",
        coordinate_type="input_output_channel",
        data_type="weights"
    )
    
    original_id = weight.id
    
    # Update the weight with new data (simulating reload)
    weight.data = weight_data["data"]
    weight.shape = weight_data["shape"]
    weight.save()
    
    # Verify the weight was updated, not recreated
    updated_weight = Weight.objects.get(coordinate="layers.0.out_0.in_0")
    assert updated_weight.id == original_id  # Same object
    assert updated_weight.data == weight_data["data"]  # New data
    assert updated_weight.shape == [3, 3]  # New shape


@pytest.mark.django_db
def test_weight_data_types():
    """Test that both weights and bias data types work correctly."""
    model = Model.objects.create(alias="test", name="Test", definition={})
    
    # Create kernel weight
    kernel_weight = Weight.objects.create(
        model=model,
        coordinate="layer.out_0.in_0",
        data=[[0.1, 0.2], [0.3, 0.4]],
        shape=[2, 2],
        layer_type="Conv2d",
        coordinate_type="input_output_channel",
        data_type="weights"
    )
    
    # Create bias weight
    bias_weight = Weight.objects.create(
        model=model,
        coordinate="layer.out_0.bias",
        data=0.5,
        shape=[],
        layer_type="Conv2d",
        coordinate_type="output_channel_bias",
        data_type="bias"
    )
    
    # Test queries
    weights_only = Weight.objects.filter(data_type="weights")
    bias_only = Weight.objects.filter(data_type="bias")
    
    assert kernel_weight in weights_only
    assert bias_weight not in weights_only
    assert bias_weight in bias_only
    assert kernel_weight not in bias_only