"""Tests for UI graph transformer functionality."""
import pytest
import networkx as nx
from neural_data.types import (
    Conv2dOutputCoordinate,
    Conv2dSliceCoordinate, 
    Conv2dInputCoordinate,
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
)
from neural_data.ui_graph_types import Conv2dInputPatchNode
from neural_data.graph.ui_graph_transformer import transform_raw_graph_to_ui_graph


@pytest.fixture
def sample_conv_output():
    """Create a sample Conv2d output coordinate."""
    return Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )


@pytest.fixture  
def sample_conv_slice():
    """Create a sample Conv2d slice coordinate."""
    return Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d", 
        coordinate_type="slice",
        in_channel=0,
        out_channel=0,
        y=5,
        x=5
    )


@pytest.fixture
def sample_conv_input_coords():
    """Create sample Conv2d input coordinates forming a 3x3 patch."""
    coords = []
    for y in range(4, 7):  # 3x3 patch centered at (5,5)
        for x in range(4, 7):
            coords.append(Conv2dInputCoordinate(
                type="Conv2dInputCoordinate",
                layer_name="conv1",
                layer_type="conv2d",
                coordinate_type="input", 
                channel=0,
                y=y,
                x=x
            ))
    return coords


@pytest.fixture
def sample_relu_input():
    """Create a sample ReLU input coordinate."""
    return ReLUInputCoordinate(
        type="ReLUInputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="input",
        channel=0,
        y=4,
        x=4
    )


def test_transform_conv2d_slice_to_patch_node(sample_conv_slice, sample_conv_input_coords):
    """Test transformation of Conv2dSliceCoordinate + Conv2dInputCoordinates to Conv2dInputPatchNode."""
    # Create raw graph
    raw_graph = nx.DiGraph()
    raw_graph.add_node(sample_conv_slice)
    
    for input_coord in sample_conv_input_coords:
        raw_graph.add_node(input_coord)
        raw_graph.add_edge(sample_conv_slice, input_coord)  # slice -> input (child -> parent)
    
    # Transform to UI graph
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    # Verify structure
    assert len(ui_graph.nodes()) == 1, "Should have single patch node"
    
    patch_node = list(ui_graph.nodes())[0]
    assert isinstance(patch_node, Conv2dInputPatchNode)
    assert patch_node.type == "Conv2dInputPatchNode"
    
    # Verify patch properties
    assert patch_node.layer_name == "conv1"
    assert patch_node.in_channel == 0
    assert patch_node.out_channel == 0
    assert patch_node.patch_min_y == 4
    assert patch_node.patch_max_y == 6 
    assert patch_node.patch_min_x == 4
    assert patch_node.patch_max_x == 6
    
    # Verify input coordinates are preserved
    assert len(patch_node.input_coordinates) == 9  # 3x3 patch
    assert set(patch_node.input_coordinates) == set(sample_conv_input_coords)


def test_pass_through_node_types(sample_conv_output, sample_relu_input):
    """Test that non-transformed node types pass through unchanged."""
    # Create raw graph with pass-through nodes
    raw_graph = nx.DiGraph()
    raw_graph.add_node(sample_conv_output)
    raw_graph.add_node(sample_relu_input) 
    raw_graph.add_edge(sample_conv_output, sample_relu_input)  # output -> relu_input (parent -> child)
    
    # Transform
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    # Verify nodes pass through unchanged
    assert len(ui_graph.nodes()) == 2
    ui_nodes = list(ui_graph.nodes())
    
    assert sample_conv_output in ui_nodes
    assert sample_relu_input in ui_nodes
    
    # Verify edge is preserved
    assert len(ui_graph.edges()) == 1
    assert ui_graph.has_edge(sample_conv_output, sample_relu_input)


def test_complex_graph_transformation(sample_conv_output, sample_conv_slice, sample_conv_input_coords, sample_relu_input):
    """Test transformation of a complex graph with multiple node types."""
    # Create complex raw graph
    raw_graph = nx.DiGraph() 
    
    # Add all nodes
    raw_graph.add_node(sample_conv_output)
    raw_graph.add_node(sample_conv_slice) 
    raw_graph.add_node(sample_relu_input)
    for input_coord in sample_conv_input_coords:
        raw_graph.add_node(input_coord)
    
    # Add edges: child -> parent direction
    for input_coord in sample_conv_input_coords:
        raw_graph.add_edge(sample_conv_slice, input_coord)  # slice -> input (child -> parent)
    raw_graph.add_edge(sample_conv_output, sample_conv_slice)  # output -> slice (child -> parent)
    
    # Connect relu as next layer (relu follows some other path)
    raw_graph.add_edge(sample_relu_input, sample_conv_input_coords[0])  # relu -> input (child -> parent)
    
    # Transform
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    # Should have: conv_output, patch_node, relu_input
    assert len(ui_graph.nodes()) == 3
    
    # Find patch node
    patch_node = None
    for node in ui_graph.nodes():
        if isinstance(node, Conv2dInputPatchNode):
            patch_node = node
            break
    
    assert patch_node is not None
    
    # Verify relationships are preserved (child -> parent direction)
    assert ui_graph.has_edge(sample_conv_output, patch_node)  # output -> patch (child -> parent)
    assert ui_graph.has_edge(sample_relu_input, patch_node)   # relu -> patch (child -> parent)


def test_empty_graph():
    """Test transformation of empty graph."""
    raw_graph = nx.DiGraph()
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    assert len(ui_graph.nodes()) == 0
    assert len(ui_graph.edges()) == 0


def test_single_node_pass_through():
    """Test transformation of graph with single pass-through node."""
    model_input = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=10,
        x=10
    )
    
    raw_graph = nx.DiGraph()
    raw_graph.add_node(model_input)
    
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    assert len(ui_graph.nodes()) == 1
    assert model_input in ui_graph.nodes()


def test_no_duplicate_edges():
    """Test that duplicate edges aren't created when multiple raw nodes map to same UI node."""
    # Create scenario where multiple input coordinates could create duplicate edges
    slice1 = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice", 
        in_channel=0,
        out_channel=0,
        y=5,
        x=5
    )
    
    input_coord = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="conv1", 
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=5,
        x=5
    )
    
    relu_input = ReLUInputCoordinate(
        type="ReLUInputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="input",
        channel=0,
        y=5, 
        x=5
    )
    
    raw_graph = nx.DiGraph()
    raw_graph.add_node(slice1)
    raw_graph.add_node(input_coord) 
    raw_graph.add_node(relu_input)
    
    raw_graph.add_edge(slice1, input_coord)  # slice -> input (child -> parent)
    raw_graph.add_edge(relu_input, input_coord)  # relu_input -> input (child -> parent)
    
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    # Should have patch node and relu input, connected by single edge
    assert len(ui_graph.nodes()) == 2
    assert len(ui_graph.edges()) == 1
    
    patch_node = None
    for node in ui_graph.nodes():
        if isinstance(node, Conv2dInputPatchNode):
            patch_node = node
            break
    
    assert patch_node is not None
    assert ui_graph.has_edge(relu_input, patch_node)  # relu -> patch (child -> parent)


def test_relu_output_removal():
    """Test that ReLU output nodes are removed from UI graphs."""
    relu_input = ReLUInputCoordinate(
        type="ReLUInputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="input",
        channel=0,
        y=0,
        x=0
    )
    
    relu_output = ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name="relu1", 
        layer_type="relu",
        coordinate_type="output",
        channel=0,
        y=0,
        x=0
    )
    
    raw_graph = nx.DiGraph()
    raw_graph.add_node(relu_input)
    raw_graph.add_node(relu_output)
    raw_graph.add_edge(relu_input, relu_output)  # input -> output
    
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    # Should only have ReLU input, ReLU output should be removed
    assert len(ui_graph.nodes()) == 1
    assert relu_input in ui_graph.nodes()
    assert relu_output not in ui_graph.nodes()
    assert len(ui_graph.edges()) == 0


def test_slice_coordinate_validation_errors():
    """Test that proper exceptions are raised for invalid slice coordinate setups."""
    # Test 1: Conv2dSliceCoordinate with no Conv2dInputCoordinate parents
    slice_coord = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=0,
        y=5,
        x=5
    )
    
    raw_graph = nx.DiGraph()
    raw_graph.add_node(slice_coord)
    
    with pytest.raises(ValueError, match="has no Conv2dInputCoordinate parents"):
        transform_raw_graph_to_ui_graph(raw_graph)
    
    # Test 2: Conv2dSliceCoordinate with mismatched layer names
    input_coord_wrong_layer = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="wrong_layer",  # Different layer
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=5,
        x=5
    )
    
    raw_graph2 = nx.DiGraph()
    raw_graph2.add_node(slice_coord)
    raw_graph2.add_node(input_coord_wrong_layer)
    raw_graph2.add_edge(slice_coord, input_coord_wrong_layer)
    
    with pytest.raises(ValueError, match="has layer_name.*but slice coordinate.*expects layer_name"):
        transform_raw_graph_to_ui_graph(raw_graph2)
    
    # Test 3: Conv2dSliceCoordinate with mismatched channels
    input_coord_wrong_channel = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate", 
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="input",
        channel=1,  # Different channel
        y=5,
        x=5
    )
    
    raw_graph3 = nx.DiGraph()
    raw_graph3.add_node(slice_coord)
    raw_graph3.add_node(input_coord_wrong_channel)
    raw_graph3.add_edge(slice_coord, input_coord_wrong_channel)
    
    with pytest.raises(ValueError, match="has channel.*but slice coordinate.*expects channel"):
        transform_raw_graph_to_ui_graph(raw_graph3)