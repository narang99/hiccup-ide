"""Transform-during-construction graph builder for neural network coordinates.

This module implements a single-phase graph construction approach that applies
transformations as the dependency graph is being built, replacing the two-phase
approach of building raw graphs and then transforming them.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Set, Callable
import networkx as nx

from neural_data.types import (
    Coordinate, 
    Conv2dSliceCoordinate, 
    Conv2dInputCoordinate, 
    Conv2dOutputCoordinate,
    ReLUOutputCoordinate, 
    ReLUInputCoordinate,
    ModelInputCoordinate
)
from neural_data.model_spec import ModelDefinition
from neural_data.ui_graph_types import Conv2dInputPatchNode
from neural_data.graph.parent_coordinates import (
    get_parents_of_conv2d_output_coordinate,
    get_parents_of_conv2d_slice_coordinate,
    get_parents_of_conv2d_input_coordinate,
    get_parents_of_relu_input_coordinate,
    get_parents_of_relu_output_coordinate,
    get_parents_of_model_input_coordinate,
)


class TransformResult(Enum):
    """Result type for transformer operations."""
    CONTINUE = "continue"      # Continue traversal with returned parents
    TERMINATE = "terminate"    # Stop traversal at this branch


class GraphTransformer(ABC):
    """Abstract base class for graph transformers."""
    
    @abstractmethod
    def transform(self, node: Coordinate, parents: List[Coordinate], builder: 'TransformingGraphBuilder') -> tuple[TransformResult, List[Coordinate]]:
        """Transform parent list for a given node.
        
        Args:
            node: The current node being processed
            parents: List of parent coordinates from raw model definition
            builder: The graph builder instance (for recursive parent queries)
            
        Returns:
            Tuple of (result_type, transformed_parents):
            - CONTINUE: Continue traversal with transformed_parents
            - TERMINATE: Stop traversal at this branch (ignore transformed_parents)
        """
        pass


class TransformingGraphBuilder:
    """Graph builder that applies transformations during construction."""
    
    def __init__(self, model_dfn: ModelDefinition):
        self.model_dfn = model_dfn
        self.transformers: List[GraphTransformer] = []
        self._cache: Dict[Coordinate, List[Coordinate]] = {}
        
        # Mapping from coordinate type to parent-finding function
        self._coordinate_to_parent_func: Dict[
            str, Callable[[Coordinate, ModelDefinition], List[Coordinate]]
        ] = {
            "Conv2dOutputCoordinate": get_parents_of_conv2d_output_coordinate,
            "Conv2dSliceCoordinate": get_parents_of_conv2d_slice_coordinate,
            "Conv2dInputCoordinate": get_parents_of_conv2d_input_coordinate,
            "ReLUInputCoordinate": get_parents_of_relu_input_coordinate,
            "ReLUOutputCoordinate": get_parents_of_relu_output_coordinate,
            "ModelInputCoordinate": get_parents_of_model_input_coordinate,
        }
    
    def add_transformer(self, transformer: GraphTransformer):
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
    
    def _get_raw_parents(self, node: Coordinate) -> List[Coordinate]:
        """Get raw parents from model definition using existing parent functions."""
        # UI nodes (like Conv2dInputPatchNode) are terminal - they don't have parents
        if hasattr(node, 'input_coordinates'):  # This is a UI patch node
            return []
        
        if node.type not in self._coordinate_to_parent_func:
            raise ValueError(f"Unsupported coordinate type: {node.type}")
        
        parent_func = self._coordinate_to_parent_func[node.type]
        return parent_func(node, self.model_dfn)
    
    def get_transformed_parents(self, node: Coordinate) -> List[Coordinate]:
        """Get transformed parents, applying transformation pipeline."""
        if node in self._cache:
            return self._cache[node]
        
        # Get raw parents from model definition
        raw_parents = self._get_raw_parents(node)
        
        # Apply transformation pipeline
        current_parents = raw_parents
        for transformer in self.transformers:
            result_type, transformed = transformer.transform(node, current_parents, self)
            
            if result_type == TransformResult.TERMINATE:
                self._cache[node] = []  # No parents - branch terminated
                return []
            
            current_parents = transformed
        
        self._cache[node] = current_parents
        return current_parents
    
    def build_graph(self, start_coordinate: Coordinate) -> nx.DiGraph:
        """Build complete transformed graph from start coordinate.
        
        Args:
            start_coordinate: The starting coordinate to build the graph from
            
        Returns:
            NetworkX directed graph where:
            - Nodes are coordinate objects (possibly transformed)
            - Edges point from child coordinates to parent coordinates
        """
        graph = nx.DiGraph()
        visited: Set[Coordinate] = set()
        self._build_recursive(start_coordinate, visited, graph)
        return graph
    
    def _build_recursive(self, coord: Coordinate, visited: Set[Coordinate], graph: nx.DiGraph) -> None:
        """Recursively build the graph with transformations applied during construction."""
        # Skip if already visited (prevents cycles)
        if coord in visited:
            return
        
        # Mark as visited and add to graph
        visited.add(coord)
        graph.add_node(coord)
        
        # Get transformed parents
        parents = self.get_transformed_parents(coord)
        
        # Recursively process parents
        for parent_coord in parents:
            # Recursively process parent
            self._build_recursive(parent_coord, visited, graph)
            
            # Add edge from current coordinate to parent (child -> parent direction)
            if parent_coord in graph:
                graph.add_edge(coord, parent_coord)


# Concrete Transformer Implementations

class Conv2dPatchMerger(GraphTransformer):
    """Transformer that consolidates Conv2d receptive fields into patch nodes."""
    
    def transform(self, node: Coordinate, parents: List[Coordinate], builder: 'TransformingGraphBuilder') -> tuple[TransformResult, List[Coordinate]]:
        """Transform Conv2dSliceCoordinate -> Conv2dInputCoordinate relationships into patch nodes."""
        if isinstance(node, Conv2dSliceCoordinate):
            # Filter for Conv2dInputCoordinate parents (the receptive field)
            input_coords = [p for p in parents if isinstance(p, Conv2dInputCoordinate)]
            
            if input_coords:
                # Create a single patch node representing the entire receptive field
                patch_node = self._create_patch_node(node, input_coords)
                
                # Return other parent types unchanged + the new patch node
                other_parents = [p for p in parents if not isinstance(p, Conv2dInputCoordinate)]
                return TransformResult.CONTINUE, other_parents + [patch_node]
        
        return TransformResult.CONTINUE, parents
    
    def _create_patch_node(self, slice_coord: Conv2dSliceCoordinate, input_coords: List[Conv2dInputCoordinate]) -> Conv2dInputPatchNode:
        """Create a Conv2dInputPatchNode from a slice and its input coordinates."""
        if not input_coords:
            raise ValueError("Cannot create patch node from empty input coordinates")
        
        # Calculate patch boundaries
        min_y = min(coord.y for coord in input_coords)
        max_y = max(coord.y for coord in input_coords)
        min_x = min(coord.x for coord in input_coords)
        max_x = max(coord.x for coord in input_coords)
        
        # All input coordinates should have the same channel (in_channel of the slice)
        in_channel = input_coords[0].channel
        if not all(coord.channel == in_channel for coord in input_coords):
            raise ValueError("All input coordinates in patch must have same channel")
        
        return Conv2dInputPatchNode(
            type="Conv2dInputPatchNode",
            layer_name=slice_coord.layer_name,
            layer_type="conv2d",
            coordinate_type="input_patch",
            in_channel=slice_coord.in_channel,
            out_channel=slice_coord.out_channel,
            patch_min_y=min_y,
            patch_min_x=min_x,
            patch_max_y=max_y,
            patch_max_x=max_x,
            input_coordinates=tuple(input_coords)
        )


class ReLUOutputSkipper(GraphTransformer):
    """Transformer that skips ReLU output coordinates to simplify graph structure."""
    
    def transform(self, node: Coordinate, parents: List[Coordinate], builder: 'TransformingGraphBuilder') -> tuple[TransformResult, List[Coordinate]]:
        """Skip ReLU output coordinates by replacing them with their parents."""
        new_parents = []
        
        for parent in parents:
            if isinstance(parent, ReLUOutputCoordinate):
                # Skip ReLU output, get its transformed parents instead
                relu_parents = builder.get_transformed_parents(parent)
                new_parents.extend(relu_parents)
            else:
                new_parents.append(parent)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_parents = []
        for parent in new_parents:
            if parent not in seen:
                seen.add(parent)
                unique_parents.append(parent)
        
        return TransformResult.CONTINUE, unique_parents


class BranchTerminator(GraphTransformer):
    """Transformer that terminates traversal at specified coordinate types."""
    
    def __init__(self, termination_types: List[type]):
        """Initialize with list of coordinate types that should terminate traversal.
        
        Args:
            termination_types: List of coordinate classes where traversal should stop
        """
        self.termination_types = tuple(termination_types)
    
    def transform(self, node: Coordinate, parents: List[Coordinate], builder: 'TransformingGraphBuilder') -> tuple[TransformResult, List[Coordinate]]:
        """Terminate traversal if node type should stop traversal, or filter terminated parent types."""
        # Terminate if this node type should stop traversal
        if isinstance(node, self.termination_types):
            return TransformResult.TERMINATE, []
        
        # Filter out parent types that should terminate traversal
        filtered_parents = [
            p for p in parents 
            if not isinstance(p, self.termination_types)
        ]
        
        return TransformResult.CONTINUE, filtered_parents


# Coordinate String Parser

def parse_coordinate_string(coord_string: str, layer_name: str, layer_type: str, y: int = 0, x: int = 0) -> Coordinate:
    """Parse coordinate string back to Coordinate object.
    
    Args:
        coord_string: String like "layers.0.out_0.in_1" or "layers.0.out_0"
        layer_name: Layer name from the model (e.g., "layers.0")
        layer_type: Layer type ("Conv2d", "ReLU", "Input")
        y: Y coordinate (default 0 for demo)
        x: X coordinate (default 0 for demo)
        
    Returns:
        Parsed Coordinate object
    """
    parts = coord_string.split('.')
    
    if layer_type.lower() == "conv2d":
        # Check if this is a slice coordinate: layers.0.out_0.in_1
        if len(parts) >= 4 and parts[-2].startswith('out_') and parts[-1].startswith('in_'):
            out_channel = int(parts[-2].split('_')[1])
            in_channel = int(parts[-1].split('_')[1])
            return Conv2dSliceCoordinate(
                type="Conv2dSliceCoordinate",
                layer_name=layer_name,
                layer_type="conv2d",
                coordinate_type="slice",
                in_channel=in_channel,
                out_channel=out_channel,
                y=y,
                x=x
            )
        # Check if this is an output coordinate: layers.0.out_0
        elif 'out_' in coord_string:
            out_channel = int(coord_string.split('out_')[1].split('.')[0])
            return Conv2dOutputCoordinate(
                type="Conv2dOutputCoordinate",
                layer_name=layer_name,
                layer_type="conv2d",
                coordinate_type="output",
                channel=out_channel,
                y=y,
                x=x
            )
    
    elif layer_type.lower() == "relu":
        if 'out_' in coord_string:
            out_channel = int(coord_string.split('out_')[1].split('.')[0])
            return ReLUOutputCoordinate(
                type="ReLUOutputCoordinate",
                layer_name=layer_name,
                layer_type="relu",
                coordinate_type="output",
                channel=out_channel,
                y=y,
                x=x
            )
    
    elif layer_type.lower() == "input":
        if 'out_' in coord_string:
            out_channel = int(coord_string.split('out_')[1].split('.')[0])
            return ModelInputCoordinate(
                type="ModelInputCoordinate",
                layer_name=layer_name,
                layer_type="input",
                channel=out_channel,
                y=y,
                x=x
            )
    
    # Fallback for unsupported layer types like Linear, Flatten - skip gracefully
    raise ValueError(f"Unsupported layer type for graph building: {layer_type}. Only Conv2d, ReLU, and Input are currently supported.")