"""Tests for the transform-during-construction graph builder."""

import pytest
import os
import networkx as nx
from neural_data.graph.transforming_builder import (
    TransformingGraphBuilder,
    GraphTransformer,
    TransformResult,
    Conv2dPatchMerger,
    ReLUOutputSkipper,
    BranchTerminator,
    parse_coordinate_string,
)
from neural_data.types import (
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
    Conv2dOutputCoordinate,
    ReLUOutputCoordinate,
    ReLUInputCoordinate,
    ModelInputCoordinate,
)
from neural_data.ui_graph_types import Conv2dInputPatchNode
from neural_data.model_spec import (
    ModelDefinition,
    Conv2dNode,
    Conv2dParams,
    ReLUNode,
    ReLUParams,
    InputNode,
    InputParams,
    ModelEdge,
)
from .helpers import MODEL_PT_PATH as MODEL_PATH, INPUT_PT_PATH as INPUT_PATH


# Mock transformer for testing core framework
class MockTransformer(GraphTransformer):
    def __init__(self, transform_func=None):
        self.transform_func = transform_func or (lambda node, parents, builder: (TransformResult.CONTINUE, parents))
        self.calls = []
    
    def transform(self, node, parents, builder):
        self.calls.append((node, parents))
        return self.transform_func(node, parents, builder)


@pytest.fixture
def simple_model_definition():
    """Create a simple model for testing."""
    return ModelDefinition(
        nodes=[
            InputNode(
                id="input",
                shape=[1, 3, 28, 28],
                type="Input",
                params=InputParams(output_shape=[1, 3, 28, 28])
            ),
            Conv2dNode(
                id="conv1",
                shape=[1, 16, 26, 26],
                type="Conv2d",
                params=Conv2dParams(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=[3, 3],
                    stride=[1, 1],
                    padding=[0, 0],
                    input_shape=[1, 3, 28, 28],
                    output_shape=[1, 16, 26, 26]
                )
            ),
            ReLUNode(
                id="relu1",
                shape=[1, 16, 26, 26],
                type="ReLU",
                params=ReLUParams(
                    input_shape=[1, 16, 26, 26],
                    output_shape=[1, 16, 26, 26]
                )
            ),
        ],
        edges=[
            ModelEdge(source="input", target="conv1"),
            ModelEdge(source="conv1", target="relu1"),
        ]
    )


def test_transforming_graph_builder_initialization():
    """Test that TransformingGraphBuilder initializes correctly."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    assert builder.model_dfn == model_def
    assert builder.transformers == []
    assert builder._cache == {}


def test_add_transformer():
    """Test adding transformers to the builder."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    transformer1 = MockTransformer()
    transformer2 = MockTransformer()
    
    builder.add_transformer(transformer1)
    builder.add_transformer(transformer2)
    
    assert len(builder.transformers) == 2
    assert builder.transformers[0] == transformer1
    assert builder.transformers[1] == transformer2


def test_transformer_pipeline():
    """Test that transformers are applied in order."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    # Mock coordinate and parents
    coord = ModelInputCoordinate(type="ModelInputCoordinate", layer_name="input", layer_type="input", channel=0, y=10, x=10)
    
    # Create transformers that modify the parent list
    def transform1(node, parents, builder):
        return TransformResult.CONTINUE, parents + ["added_by_transformer1"]
    
    def transform2(node, parents, builder):
        return TransformResult.CONTINUE, parents + ["added_by_transformer2"]
    
    transformer1 = MockTransformer(transform1)
    transformer2 = MockTransformer(transform2)
    
    builder.add_transformer(transformer1)
    builder.add_transformer(transformer2)
    
    # Mock the raw parent function to return empty list
    builder._get_raw_parents = lambda node: []
    
    result = builder.get_transformed_parents(coord)
    
    assert result == ["added_by_transformer1", "added_by_transformer2"]
    assert len(transformer1.calls) == 1
    assert len(transformer2.calls) == 1


def test_transformer_termination():
    """Test that TERMINATE stops the transformation pipeline."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    coord = ModelInputCoordinate(type="ModelInputCoordinate", layer_name="input", layer_type="input", channel=0, y=10, x=10)
    
    # First transformer terminates
    def terminate_transform(node, parents, builder):
        return TransformResult.TERMINATE, []
    
    transformer1 = MockTransformer(terminate_transform)
    transformer2 = MockTransformer()  # Should not be called
    
    builder.add_transformer(transformer1)
    builder.add_transformer(transformer2)
    
    builder._get_raw_parents = lambda node: ["parent1"]
    
    result = builder.get_transformed_parents(coord)
    
    assert result == []  # Terminated
    assert len(transformer1.calls) == 1
    assert len(transformer2.calls) == 0  # Not called due to termination


def test_caching():
    """Test that transformed parents are cached."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    coord = ModelInputCoordinate(type="ModelInputCoordinate", layer_name="input", layer_type="input", channel=0, y=10, x=10)
    
    transformer = MockTransformer()
    builder.add_transformer(transformer)
    
    # Mock raw parents function
    call_count = 0
    def mock_get_raw_parents(node):
        nonlocal call_count
        call_count += 1
        return ["parent"]
    
    builder._get_raw_parents = mock_get_raw_parents
    
    # First call
    result1 = builder.get_transformed_parents(coord)
    # Second call should use cache
    result2 = builder.get_transformed_parents(coord)
    
    assert result1 == result2 == ["parent"]
    assert call_count == 1  # Raw function called only once
    assert len(transformer.calls) == 1  # Transformer called only once


def test_conv2d_patch_merger():
    """Test Conv2dPatchMerger transformer."""
    merger = Conv2dPatchMerger()
    
    # Create test coordinates
    slice_coord = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=1,
        y=10,
        x=10
    )
    
    input_coord1 = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="input",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=8,
        x=8
    )
    
    input_coord2 = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="input",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=9,
        x=9
    )
    
    other_coord = ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name="relu",
        layer_type="relu",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )
    
    parents = [input_coord1, input_coord2, other_coord]
    
    result_type, transformed_parents = merger.transform(slice_coord, parents, None)
    
    assert result_type == TransformResult.CONTINUE
    assert len(transformed_parents) == 2  # patch node + other_coord
    
    # Find the patch node
    patch_node = None
    for parent in transformed_parents:
        if isinstance(parent, Conv2dInputPatchNode):
            patch_node = parent
            break
    
    assert patch_node is not None
    assert patch_node.type == "Conv2dInputPatchNode"
    assert patch_node.patch_min_y == 8
    assert patch_node.patch_max_y == 9
    assert patch_node.patch_min_x == 8
    assert patch_node.patch_max_x == 9
    assert len(patch_node.input_coordinates) == 2
    assert other_coord in transformed_parents


def test_relu_output_skipper():
    """Test ReLUOutputSkipper transformer."""
    skipper = ReLUOutputSkipper()
    
    # Create test coordinates
    node = Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="output",
        channel=0,
        y=10,
        x=10
    )
    
    relu_output = ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )
    
    other_coord = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=2,
        x=2
    )
    
    parents = [relu_output, other_coord]
    
    # Mock builder to return parents for ReLU output
    class MockBuilder:
        def get_transformed_parents(self, coord):
            if coord == relu_output:
                return [other_coord]  # ReLU's parents
            return []
    
    mock_builder = MockBuilder()
    
    result_type, transformed_parents = skipper.transform(node, parents, mock_builder)
    
    assert result_type == TransformResult.CONTINUE
    # Should skip ReLU output and include its parent + the other parent (deduplicated)
    assert len(transformed_parents) == 1
    assert other_coord in transformed_parents
    assert relu_output not in transformed_parents


def test_branch_terminator():
    """Test BranchTerminator transformer."""
    terminator = BranchTerminator([ModelInputCoordinate])
    
    # Test termination of specific node type
    model_input = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=10,
        x=10
    )
    
    result_type, transformed_parents = terminator.transform(model_input, [], None)
    assert result_type == TransformResult.TERMINATE
    assert transformed_parents == []
    
    # Test filtering of parent types
    other_coord = Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )
    
    parents = [model_input, other_coord]
    
    result_type, transformed_parents = terminator.transform(other_coord, parents, None)
    assert result_type == TransformResult.CONTINUE
    assert len(transformed_parents) == 1
    assert other_coord in transformed_parents
    assert model_input not in transformed_parents


@pytest.mark.django_db
def test_build_graph_basic(simple_model_definition):
    """Test basic graph building functionality."""
    builder = TransformingGraphBuilder(simple_model_definition)
    
    # Create a simple coordinate
    coord = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=10,
        x=10
    )
    
    # Mock the parent function to prevent actual parent finding
    builder._get_raw_parents = lambda node: []
    
    graph = builder.build_graph(coord)
    
    assert isinstance(graph, nx.DiGraph)
    assert coord in graph.nodes()


def test_integration_with_transformers():
    """Test integration of multiple transformers in a realistic scenario."""
    model_def = ModelDefinition(nodes=[], edges=[])
    builder = TransformingGraphBuilder(model_def)
    
    # Add transformers in realistic order
    builder.add_transformer(Conv2dPatchMerger())
    builder.add_transformer(ReLUOutputSkipper())
    builder.add_transformer(BranchTerminator([ModelInputCoordinate]))
    
    # Mock coordinate that would be affected by all transformers
    slice_coord = Conv2dSliceCoordinate(
        type="Conv2dSliceCoordinate",
        layer_name="conv1",
        layer_type="conv2d",
        coordinate_type="slice",
        in_channel=0,
        out_channel=1,
        y=10,
        x=10
    )
    
    input_coord = Conv2dInputCoordinate(
        type="Conv2dInputCoordinate",
        layer_name="input",
        layer_type="conv2d",
        coordinate_type="input",
        channel=0,
        y=8,
        x=8
    )
    
    relu_output = ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name="relu1",
        layer_type="relu",
        coordinate_type="output",
        channel=0,
        y=5,
        x=5
    )
    
    model_input = ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name="input",
        layer_type="input",
        channel=0,
        y=2,
        x=2
    )
    
    # Mock parent functions
    def mock_get_parents(node):
        if node == slice_coord:
            return [input_coord, relu_output]
        elif node == relu_output:
            return [model_input]
        return []
    
    builder._get_raw_parents = mock_get_parents
    
    transformed_parents = builder.get_transformed_parents(slice_coord)
    
    # Should have patch node (from Conv2dPatchMerger)
    # ReLU output gets skipped by ReLUOutputSkipper, which resolves to ModelInputCoordinate
    # But BranchTerminator filters out ModelInputCoordinate, so only patch node remains
    assert len(transformed_parents) == 1
    
    patch_nodes = [p for p in transformed_parents if isinstance(p, Conv2dInputPatchNode)]
    model_inputs = [p for p in transformed_parents if isinstance(p, ModelInputCoordinate)]
    
    assert len(patch_nodes) == 1
    assert len(model_inputs) == 0  # Filtered out by BranchTerminator


def test_coordinate_parser():
    """Test coordinate string parsing functionality."""
    # Test Conv2d slice coordinate
    coord1 = parse_coordinate_string("layers.0.out_0.in_1", "layers.0", "Conv2d", y=5, x=3)
    assert isinstance(coord1, Conv2dSliceCoordinate)
    assert coord1.layer_name == "layers.0"
    assert coord1.out_channel == 0
    assert coord1.in_channel == 1
    assert coord1.y == 5
    assert coord1.x == 3
    
    # Test Conv2d output coordinate
    coord2 = parse_coordinate_string("layers.0.out_2", "layers.0", "Conv2d", y=1, x=2)
    assert isinstance(coord2, Conv2dOutputCoordinate)
    assert coord2.layer_name == "layers.0"
    assert coord2.channel == 2
    assert coord2.y == 1
    assert coord2.x == 2
    
    # Test ReLU output coordinate
    coord3 = parse_coordinate_string("layers.1.out_3", "layers.1", "ReLU")
    assert isinstance(coord3, ReLUOutputCoordinate)
    assert coord3.layer_name == "layers.1"
    assert coord3.channel == 3
    
    # Test Input coordinate
    coord4 = parse_coordinate_string("x.out_0", "x", "Input")
    assert isinstance(coord4, ModelInputCoordinate)
    assert coord4.layer_name == "x"
    assert coord4.channel == 0
    
    # Test invalid coordinate
    with pytest.raises(ValueError):
        parse_coordinate_string("invalid", "layer", "Unknown")


# Integration Tests with Real Model Data

def requires_real_model():
    """Decorator to skip tests if real model files are not available"""
    return pytest.mark.skipif(
        not (os.path.exists(MODEL_PATH) and os.path.exists(INPUT_PATH)),
        reason="Real model and input files not found",
    )


@pytest.fixture
def real_model_data():
    """Create test data that matches the actual model architecture using loader utilities"""
    from neural_data.load_folder.model_loader import load_model
    from neural_data.load_folder.input_loader import load_inputs
    from io import StringIO
    from pathlib import Path
    from neural_data.models import Input, SaliencyMap

    stdout = StringIO()
    # MODEL_PATH is /.../pt-to-api/data/model.pt
    # folder_path should be /.../pt-to-api/
    folder_path = Path(MODEL_PATH).parent.parent
    
    model_info = {
        "name": "real-mnist-model-for-graph-test",
        "path": "data/model.pt"
    }
    
    model_obj, model_pt_path = load_model(model_info, folder_path, stdout)
    
    input_info = {
        "path": "data/first-input-tens.pt",
        "label": 0
    }
    
    load_inputs([input_info], model_obj, folder_path, model_pt_path, stdout)
    
    # The loader uses the filename stem as alias, so 'first-input-tens.pt' -> 'first-input-tens'
    input_obj = Input.objects.get(model=model_obj, alias="first-input-tens")
    saliency_maps = SaliencyMap.objects.filter(input=input_obj)

    return model_obj, input_obj, saliency_maps


@pytest.mark.django_db
@requires_real_model()
def test_transforming_builder_with_real_model_data(real_model_data):
    """Test the transforming builder with real model data and coordinates."""
    model_obj, input_obj, saliency_maps = real_model_data
    
    # Get the real model definition from the loaded model and parse it
    from neural_data.model_spec import ModelDefinition
    model_definition_dict = model_obj.definition
    model_definition = ModelDefinition.model_validate(model_definition_dict)
    
    # Create builder with real model definition
    builder = TransformingGraphBuilder(model_definition)
    
    # Add realistic transformers
    builder.add_transformer(Conv2dPatchMerger())
    builder.add_transformer(ReLUOutputSkipper())
    builder.add_transformer(BranchTerminator([ModelInputCoordinate]))
    
    # Pick a real coordinate from earlier layers (Conv2d/ReLU layers)
    target_layers = ['layers.3', 'layers.2', 'layers.1', 'layers.0']
    sample_saliency = None
    sample_activation = None
    
    # Find a saliency map from one of the supported layer types
    for layer_name in target_layers:
        sample_saliency = saliency_maps.filter(layer_name=layer_name).first()
        if sample_saliency:
            from neural_data.models import Activation
            sample_activation = Activation.objects.filter(
                input=input_obj, 
                coordinate=sample_saliency.coordinate,
                layer_name=layer_name
            ).first()
            if sample_activation:
                break
    
    if sample_saliency and sample_activation:
            start_coordinate = parse_coordinate_string(
                sample_saliency.coordinate,
                sample_activation.layer_name,
                sample_activation.layer_type,
                y=0, x=0  # Using defaults for demo
            )
            
            # Build graph from real coordinate
            graph = builder.build_graph(start_coordinate)
            
            # Verify the graph was built
            assert isinstance(graph, nx.DiGraph)
            assert len(graph.nodes()) > 0
            assert start_coordinate in graph.nodes()
            
            # Check that transformations were applied by looking for UI node types
            ui_nodes = [node for node in graph.nodes() if isinstance(node, Conv2dInputPatchNode)]
            
            # The exact number depends on the model structure, but if there are Conv2d layers,
            # we should see some patch nodes
            print(f"Graph has {len(graph.nodes())} total nodes")
            print(f"Graph has {len(ui_nodes)} patch nodes")
            print(f"Node types: {[type(node).__name__ for node in graph.nodes()]}")
    else:
        pytest.skip("No activation data found for coordinate parsing")


@pytest.mark.django_db
@requires_real_model()
def test_transforming_builder_caching_with_real_data(real_model_data):
    """Test that caching works correctly with real model coordinates."""
    model_obj, input_obj, saliency_maps = real_model_data
    
    from neural_data.model_spec import ModelDefinition
    model_definition_dict = model_obj.definition
    model_definition = ModelDefinition.model_validate(model_definition_dict)
    builder = TransformingGraphBuilder(model_definition)
    builder.add_transformer(Conv2dPatchMerger())
    
    # Use a real coordinate from supported layers
    target_layers = ['layers.3', 'layers.2', 'layers.1', 'layers.0']
    sample_saliency = None
    sample_activation = None
    
    for layer_name in target_layers:
        sample_saliency = saliency_maps.filter(layer_name=layer_name).first()
        if sample_saliency:
            from neural_data.models import Activation
            sample_activation = Activation.objects.filter(
                input=input_obj, 
                coordinate=sample_saliency.coordinate,
                layer_name=layer_name
            ).first()
            if sample_activation:
                break
    
    if sample_saliency and sample_activation:
            coord = parse_coordinate_string(
                sample_saliency.coordinate,
                sample_activation.layer_name,
                sample_activation.layer_type,
                y=0, x=0
            )
            
            # First call
            parents1 = builder.get_transformed_parents(coord)
            
            # Check that cache was populated
            assert coord in builder._cache
            
            # Second call should return same result from cache
            parents2 = builder.get_transformed_parents(coord)
            
            assert parents1 == parents2
    else:
        pytest.skip("No activation data found for coordinate parsing")


@pytest.mark.django_db
@requires_real_model()
def test_different_transformer_combinations_with_real_data(real_model_data):
    """Test different combinations of transformers with real model data."""
    model_obj, input_obj, saliency_maps = real_model_data
    
    from neural_data.model_spec import ModelDefinition
    model_definition_dict = model_obj.definition
    model_definition = ModelDefinition.model_validate(model_definition_dict)
    
    # Test 1: Only patch merger
    builder1 = TransformingGraphBuilder(model_definition)
    builder1.add_transformer(Conv2dPatchMerger())
    
    # Test 2: Patch merger + ReLU skipper
    builder2 = TransformingGraphBuilder(model_definition)
    builder2.add_transformer(Conv2dPatchMerger())
    builder2.add_transformer(ReLUOutputSkipper())
    
    # Test 3: All transformers
    builder3 = TransformingGraphBuilder(model_definition)
    builder3.add_transformer(Conv2dPatchMerger())
    builder3.add_transformer(ReLUOutputSkipper())
    builder3.add_transformer(BranchTerminator([ModelInputCoordinate]))
    
    # Find coordinate from supported layers
    target_layers = ['layers.3', 'layers.2', 'layers.1', 'layers.0']
    sample_saliency = None
    sample_activation = None
    
    for layer_name in target_layers:
        sample_saliency = saliency_maps.filter(layer_name=layer_name).first()
        if sample_saliency:
            from neural_data.models import Activation
            sample_activation = Activation.objects.filter(
                input=input_obj, 
                coordinate=sample_saliency.coordinate,
                layer_name=layer_name
            ).first()
            if sample_activation:
                break
    
    if sample_saliency and sample_activation:
            coord = parse_coordinate_string(
                sample_saliency.coordinate,
                sample_activation.layer_name,
                sample_activation.layer_type,
                y=0, x=0
            )
            
            # Build graphs with different transformer combinations
            graph1 = builder1.build_graph(coord)
            graph2 = builder2.build_graph(coord)
            graph3 = builder3.build_graph(coord)
            
            # All should produce valid graphs
            for i, graph in enumerate([graph1, graph2, graph3], 1):
                assert isinstance(graph, nx.DiGraph)
                assert len(graph.nodes()) > 0
                print(f"Builder {i}: {len(graph.nodes())} nodes")
            
            # Graph 3 (with BranchTerminator) should potentially have fewer nodes
            # since it stops at ModelInputCoordinates
            print(f"Graph sizes: {len(graph1.nodes())}, {len(graph2.nodes())}, {len(graph3.nodes())}")
    else:
        pytest.skip("No activation data found for coordinate parsing")


@pytest.mark.django_db 
@requires_real_model()
def test_performance_comparison_hint(real_model_data):
    """Basic performance test to demonstrate single-phase efficiency."""
    model_obj, input_obj, saliency_maps = real_model_data
    
    from neural_data.model_spec import ModelDefinition
    model_definition_dict = model_obj.definition
    model_definition = ModelDefinition.model_validate(model_definition_dict)
    
    # Test the transforming builder (single-phase)
    builder = TransformingGraphBuilder(model_definition)
    builder.add_transformer(Conv2dPatchMerger())
    builder.add_transformer(ReLUOutputSkipper())
    builder.add_transformer(BranchTerminator([ModelInputCoordinate]))
    
    # Find coordinate from supported layers  
    target_layers = ['layers.3', 'layers.2', 'layers.1', 'layers.0']
    sample_saliency = None
    sample_activation = None
    
    for layer_name in target_layers:
        sample_saliency = saliency_maps.filter(layer_name=layer_name).first()
        if sample_saliency:
            from neural_data.models import Activation
            sample_activation = Activation.objects.filter(
                input=input_obj, 
                coordinate=sample_saliency.coordinate,
                layer_name=layer_name
            ).first()
            if sample_activation:
                break
    
    if sample_saliency and sample_activation:
            coord = parse_coordinate_string(
                sample_saliency.coordinate,
                sample_activation.layer_name,
                sample_activation.layer_type,
                y=0, x=0
            )
            
            import time
            start = time.time()
            graph = builder.build_graph(coord)
            transform_time = time.time() - start
            
            print(f"Transform-during-construction took {transform_time:.4f}s for {len(graph.nodes())} nodes")
            
            # Basic validation
            assert isinstance(graph, nx.DiGraph)
            assert len(graph.nodes()) > 0
            
            # The single-phase approach should be efficient
            # (This is more of a demonstration than a strict assertion)
            assert transform_time < 10.0  # Should complete in reasonable time
    else:
        pytest.skip("No activation data found for coordinate parsing")