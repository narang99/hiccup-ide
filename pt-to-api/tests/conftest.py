import pytest
import torch
import torch.nn as nn
from pt_to_api.capture import get_model_internals


@pytest.fixture
def test_model():
    """Create a simple test model matching the example-model.json structure."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # layers_0
                nn.ReLU(),                                            # layers_1
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # layers_2
                nn.ReLU(),                                            # layers_3
                nn.Flatten(),                                         # layers_4
                nn.Linear(784, 10)                                    # layers_5
            )
        
        def forward(self, x):
            return self.layers(x)
    
    return TestModel()


@pytest.fixture
def test_input():
    """Create test input tensor."""
    return torch.randn(1, 1, 28, 28)


@pytest.fixture(scope="module")
def model_snapshot():
    """Create model snapshot with activations and parameters (module-level fixture)."""
    # Create model and input
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(784, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    input_tensor = torch.randn(1, 1, 28, 28)
    
    # Get model internals once for the entire test module
    activations, parameters = get_model_internals(model, input_tensor)
    
    return {
        'model': model,
        'input_tensor': input_tensor,
        'activations': activations,
        'parameters': parameters
    }