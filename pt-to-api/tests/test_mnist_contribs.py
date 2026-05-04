import torch
import pytest
from pt_to_api.mnist import (
    SimpleMNIST,
    get_contribs_for_inp_vectorized,
    LayerBackpropController,
    _get_0_slice_contribs,
)
from pt_to_api.capture import get_model_internals


@pytest.fixture
def model():
    """Create a SimpleMNIST model for testing"""
    return SimpleMNIST()


@pytest.fixture
def batch_input():
    """Create a batch input tensor for testing"""
    return torch.randn(2, 1, 28, 28)


@pytest.fixture
def device():
    """Use CPU device for testing"""
    return "cpu"


class TestLayerBackpropController:
    def test_initialization(self):
        """Test LayerBackpropController initialization"""
        layer_names = [
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
        ]
        controller = LayerBackpropController(layer_names, "layers.2")

        assert controller.last_layer_key == "layers.2"
        assert controller.last_layer_index == 2
        assert "layers.0.slice" in controller.slice_dependencies
        assert "layers.2.slice" in controller.slice_dependencies

    def test_should_backprop_regular_layers(self):
        """Test should_backprop for regular layers"""
        layer_names = [
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
        ]
        controller = LayerBackpropController(layer_names, "layers.2")

        # Layers at or before last_layer should backprop
        assert controller.should_backprop("layers.0") is True
        assert controller.should_backprop("layers.1") is True
        assert controller.should_backprop("layers.2") is True
        assert controller.should_backprop("x") is True  # x is always backprop

        # Layers after last_layer should not backprop
        assert controller.should_backprop("layers.3") is False
        assert controller.should_backprop("layers.4") is False
        assert controller.should_backprop("layers.5") is False

    def test_should_backprop_slice_layers(self):
        """Test should_backprop for slice layers based on dependencies"""
        layer_names = [
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
        ]

        # Test with last_layer = layers.1
        controller = LayerBackpropController(layer_names, "layers.1")
        assert controller.should_backprop("layers.0.slice") is True  # depends on x
        assert (
            controller.should_backprop("layers.2.slice") is True
        )  # depends on layers.1, and layers.1 <= last_layer

        # Test with last_layer = layers.0
        controller = LayerBackpropController(layer_names, "layers.0")
        assert controller.should_backprop("layers.0.slice") is True  # depends on x
        assert (
            controller.should_backprop("layers.2.slice") is False
        )  # depends on layers.1, but layers.1 > last_layer

        # Test with last_layer = layers.2
        controller = LayerBackpropController(layer_names, "layers.2")
        assert controller.should_backprop("layers.0.slice") is True  # depends on x
        assert (
            controller.should_backprop("layers.2.slice") is True
        )  # depends on layers.1, and layers.1 <= last_layer


class TestContribsCalculation:
    def test_all_layer_keys_have_contribs(self, model, batch_input, device):
        """Test that all expected keys are present in contributions"""
        # Create dummy last layer contributions
        with torch.no_grad():
            model.eval()
            output = model(batch_input)
            last_layer_contribs = torch.ones_like(output)

        total_contribs, acts, parameters = get_contribs_for_inp_vectorized(
            batch_input, model, last_layer_contribs, "layers.5", device
        )

        expected_keys = [
            "x",
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
            "layers.0.slice",
            "layers.2.slice",
        ]

        for key in expected_keys:
            assert key in total_contribs, f"Missing key: {key}"

    @pytest.mark.parametrize(
        "last_layer_key",
        ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5"],
    )
    def test_contribs_are_backpropped_or_zero(
        self, model, batch_input, device, last_layer_key
    ):
        """Test that zero contributions have correct tensor shapes for all last layer cases"""
        # Get activations for shape reference
        acts, _ = get_model_internals(model, batch_input)
        acts["x"] = batch_input

        # Create dummy last layer contributions
        with torch.no_grad():
            model.eval()
            output = model(batch_input)
            if last_layer_key == "layers.5":
                last_layer_contribs = torch.ones_like(output)
            else:
                # Get the shape of the last layer activations
                last_layer_contribs = torch.ones_like(acts[last_layer_key])

        total_contribs, _, _ = get_contribs_for_inp_vectorized(
            batch_input, model, last_layer_contribs, last_layer_key, device
        )

        # Check that all contributions have correct shapes
        layer_names = [
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
            "layers.0.slice",
            "layers.2.slice",
        ]
        controller = LayerBackpropController(layer_names, last_layer_key)

        for layer_name in layer_names:
            if layer_name in acts and not controller.should_backprop(layer_name):
                assert total_contribs[layer_name].shape == acts[layer_name].shape
                are_equal_to_zero = torch.allclose(
                    total_contribs[layer_name],
                    torch.zeros_like(acts[layer_name]),
                    atol=1e-7,
                )
                if controller.should_backprop(layer_name):
                    # Verify it's actually zero (or at least very small)
                    assert not are_equal_to_zero
                else:
                    assert are_equal_to_zero

    @pytest.mark.parametrize("last_layer_key", ["layers.0", "layers.1", "layers.2"])
    def test_slice_contribs_shapes(self, model, batch_input, device, last_layer_key):
        """Test slice contributions are handled correctly based on dependencies"""
        # Get activations for shape reference
        acts, _ = get_model_internals(model, batch_input)
        acts["x"] = batch_input

        # Create dummy last layer contributions
        last_layer_contribs = torch.ones_like(acts[last_layer_key])

        total_contribs, _, _ = get_contribs_for_inp_vectorized(
            batch_input, model, last_layer_contribs, last_layer_key, device
        )

        # Check layers.0.slice (depends on x)
        assert "layers.0.slice" in total_contribs
        # Should have shape [batch, out_channels, in_channels, out_h, out_w]
        expected_shape = (
            batch_input.shape[0],
            8,
            1,
            14,
            14,
        )  # First conv: 1->8 channels, 28->14 size
        assert total_contribs["layers.0.slice"].shape == expected_shape

        # Check layers.2.slice (depends on layers.1)
        assert "layers.2.slice" in total_contribs
        # Should have shape [batch, out_channels, in_channels, out_h, out_w]
        expected_shape = (
            batch_input.shape[0],
            16,
            8,
            7,
            7,
        )  # Second conv: 8->16 channels, 14->7 size
        assert total_contribs["layers.2.slice"].shape == expected_shape


class TestHelperFunctions:
    def test_get_0_slice_contribs_shape(self):
        """Test _get_0_slice_contribs returns correct shape"""
        curr_act = torch.randn(2, 16, 7, 7)  # [batch, channels, height, width]
        prev_act = torch.randn(2, 8, 14, 14)

        slice_contribs = _get_0_slice_contribs(curr_act, prev_act)

        expected_shape = (
            2,
            16,
            8,
            7,
            7,
        )  # [batch, out_channels, in_channels, out_h, out_w]
        assert slice_contribs.shape == expected_shape
        assert torch.allclose(slice_contribs, torch.zeros_like(slice_contribs))
