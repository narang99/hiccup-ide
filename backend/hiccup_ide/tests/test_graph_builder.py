"""Basic test for graph building functionality."""

from neural_data.types import Conv2dOutputCoordinate
from neural_data.model_spec import (
    ModelDefinition, Conv2dNode, Conv2dParams, InputNode, InputParams, ModelEdge
)
from neural_data.graph.graph_builder import build_graph

def filter_no_slices(coord):
    return coord.type != "Conv2dSliceCoordinate"

def test_basic_graph_building():
    """Test basic graph building functionality with a simple model."""
    # Create a simple model definition with input -> conv2d
    input_node = InputNode(
        id="input",
        type="Input",
        params=InputParams(output_shape=[1, 3, 28, 28]),
        shape=[1, 3, 28, 28]
    )
    
    conv_node = Conv2dNode(
        id="conv1",
        type="Conv2d",
        params=Conv2dParams(
            in_channels=3,
            out_channels=16,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            input_shape=[1, 3, 28, 28],
            output_shape=[1, 16, 28, 28]
        ),
        shape=[1, 16, 28, 28]
    )
    
    model_dfn = ModelDefinition(
        nodes=[input_node, conv_node],
        edges=[ModelEdge(source="input", target="conv1")]
    )
    
    # Create a test coordinate
    test_coord = Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )
    
    # Build graph without filter
    graph = build_graph(test_coord, model_dfn)
    
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Node types in graph: {[node.type for node in graph.nodes()]}")
    
    # Test with filter that excludes slice coordinates
    filtered_graph = build_graph(test_coord, model_dfn, filter_no_slices)
    
    print(f"Filtered graph has {filtered_graph.number_of_nodes()} nodes and {filtered_graph.number_of_edges()} edges")
    print(f"Node types in filtered graph: {[node.type for node in filtered_graph.nodes()]}")