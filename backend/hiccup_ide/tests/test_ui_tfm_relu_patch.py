import pytest
import networkx as nx
from neural_data.types import (
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
    SingleConv2dOpNode,
)
from neural_data.graph.ui_tfm.slice2patch import Slice2PatchStrategy
from neural_data.graph.ui_tfm.core import Consumed, Skip


def test_slice2relu_strategy_matches_pattern():
    # Arrange
    strategy = Slice2PatchStrategy()
    raw_graph = nx.DiGraph()
    
    # Pattern: Slice -> Input -> ReLU
    slice_node = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=0,
        y=0,
        x=0,
    )
    
    input_node = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=0,
        x=0,
    )
    
    relu_node = ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="output",
        channel=0,
        y=0,
        x=0,
    )
    
    # Base input (ModelInput) as child of ReLU
    model_input = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=0,
        x=0,
    )
    
    raw_graph.add_edge(slice_node, input_node)
    raw_graph.add_edge(input_node, relu_node)
    raw_graph.add_edge(relu_node, model_input)
    
    tfm_graph = nx.DiGraph()
    cache = {}
    
    def mock_main_strategy(root, g, c, t, s):
        # When called for model_input, just return it as consumed
        return Consumed(nodes=[root])

    # Act
    result = strategy(slice_node, raw_graph, cache, tfm_graph, mock_main_strategy)

    # Assert
    assert isinstance(result, Consumed)
    assert len(result.nodes) == 1
    patch_node = result.nodes[0]
    assert isinstance(patch_node, SingleConv2dOpNode)
    assert patch_node.layer_name == "conv1"
    assert patch_node.output_slice == slice_node
    assert patch_node.input_patch.layer_name == "relu1"
    assert patch_node.input_patch.channel == 0
    assert patch_node.input_patch.patch_min_y == 0
    assert patch_node.input_patch.patch_max_y == 0
    
    # Check if patch_node was added to tfm_graph
    assert patch_node in tfm_graph.nodes
    # Check if it has edge to model_input (which was returned by mock_main_strategy)
    assert model_input in tfm_graph.successors(patch_node)


def test_slice2patch_strategy_handles_model_input():
    # Arrange
    strategy = Slice2PatchStrategy()
    raw_graph = nx.DiGraph()

    slice_node = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=0,
        y=0,
        x=0,
    )
    input_node = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=0,
        x=0,
    )

    # ModelInput successor
    model_input_node = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=0,
        x=0,
    )

    raw_graph.add_edge(slice_node, input_node)
    raw_graph.add_edge(input_node, model_input_node)

    tfm_graph = nx.DiGraph()
    cache = {}

    # Act
    result = strategy(slice_node, raw_graph, cache, tfm_graph, lambda *args, **kwargs: Consumed(nodes=[]))

    # Assert
    assert isinstance(result, Consumed)
    patch_node = result.nodes[0]
    assert isinstance(patch_node, SingleConv2dOpNode)
    assert patch_node.input_patch.layer_type == "input"


def test_slice2patch_strategy_skips_invalid_types():
    # Arrange
    strategy = Slice2PatchStrategy()
    raw_graph = nx.DiGraph()

    slice_node = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=0,
        y=0,
        x=0,
    )
    input_node = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=0,
        x=0,
    )

    # Invalid successor (e.g., another Conv2dInputCoordinate instead of ReLU or ModelInput)
    invalid_node = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="conv2",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=0,
        x=0,
    )

    raw_graph.add_edge(slice_node, input_node)
    raw_graph.add_edge(input_node, invalid_node)

    tfm_graph = nx.DiGraph()
    cache = {}

    # Act
    result = strategy(slice_node, raw_graph, cache, tfm_graph, None)

    # Assert
    assert isinstance(result, Skip)

