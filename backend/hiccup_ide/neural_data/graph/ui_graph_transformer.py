"""Graph transformer for converting raw computation graphs to UI-optimized graphs.

This module handles the transformation of raw neural network dependency graphs
into forms optimized for UI visualization, specifically consolidating receptive
field relationships into single patch nodes.
"""
from typing import Dict, Set
from enum import Enum
import networkx as nx
from neural_data.types import (
    Coordinate, 
    Conv2dSliceCoordinate, 
    Conv2dInputCoordinate,
    Conv2dOutputCoordinate,
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
)
from neural_data.ui_graph_types import UIGraphNode, Conv2dInputPatchNode


class TransformAction(Enum):
    """Actions to take for different coordinate types during transformation."""
    PASS_THROUGH = "pass_through"  # Use node as-is
    DEFER_TO_SLICE = "defer_to_slice"  # Will be handled when slice is processed
    REMOVE = "remove"  # Remove from UI graph entirely


def transform_raw_graph_to_ui_graph(raw_graph: nx.DiGraph) -> nx.DiGraph:
    """Transform a raw dependency graph into a UI-optimized graph.
    
    This transformation consolidates Conv2dSliceCoordinate -> [Conv2dInputCoordinate]
    relationships into single Conv2dInputPatchNode objects for cleaner UI visualization.
    
    Transformations:
    - Conv2dSliceCoordinate -> [Conv2dInputCoordinate] becomes Conv2dInputPatchNode
    - All other node types pass through unchanged
    - Parent-child relationships are preserved with appropriate mapping
    
    Args:
        raw_graph: NetworkX directed graph with raw coordinate nodes
        
    Returns:
        NetworkX directed graph with UI-optimized node types
    """
    ui_graph = nx.DiGraph()
    
    # Track node mappings for relationship preservation
    raw_to_ui_mapping: Dict[Coordinate, UIGraphNode] = {}
    
    # First pass: Process nodes based on their transformation requirements
    for node in raw_graph.nodes():
        action, ui_node = _get_transform_action_and_node(node, raw_graph, raw_to_ui_mapping)
        
        if action == TransformAction.PASS_THROUGH:
            # Use node as-is in UI graph
            ui_graph.add_node(ui_node)
            raw_to_ui_mapping[node] = ui_node
            
        elif action == TransformAction.DEFER_TO_SLICE:
            # Will be handled when corresponding slice coordinate is processed
            # Check if already processed (slice coordinates process their inputs)
            if node in raw_to_ui_mapping:
                ui_node = raw_to_ui_mapping[node]
                if ui_node not in ui_graph:
                    ui_graph.add_node(ui_node)
                    
        elif action == TransformAction.REMOVE:
            # Explicitly skip - node will not appear in UI graph
            pass
            
        else:
            raise ValueError(f"Unknown transform action: {action}")
    
    # Second pass: Create edges with proper parent-child relationships
    _create_ui_edges(raw_graph, ui_graph, raw_to_ui_mapping)
    
    # Final validation: Ensure no disallowed node types made it into the UI graph
    for ui_node in ui_graph.nodes():
        if isinstance(ui_node, (Conv2dSliceCoordinate, Conv2dInputCoordinate, ReLUOutputCoordinate)):
            raise ValueError(
                f"UI graph contains node {ui_node} of type {type(ui_node).__name__}, "
                f"but this type should have been transformed or removed. "
                f"This indicates a bug in the transformation logic."
            )
    
    return ui_graph


def _get_transform_action_and_node(
    node: Coordinate, 
    raw_graph: nx.DiGraph,
    raw_to_ui_mapping: Dict[Coordinate, UIGraphNode]
) -> tuple[TransformAction, UIGraphNode | None]:
    """Determine how to transform a node and return the action and resulting UI node.
    
    Returns:
        tuple of (TransformAction, UIGraphNode | None)
        - For PASS_THROUGH: (action, transformed_node)
        - For DEFER_TO_SLICE: (action, None) - node will be processed when slice is handled
        - For REMOVE: (action, None) - node will not appear in UI graph
    """
    if isinstance(node, Conv2dSliceCoordinate):
        # Transform slice + its input coordinates into a single patch node
        patch_node = _create_input_patch_node(node, raw_graph, raw_to_ui_mapping)
        return TransformAction.PASS_THROUGH, patch_node
        
    elif isinstance(node, Conv2dInputCoordinate):
        # These are handled as part of Conv2dInputPatchNode creation
        # Check if already processed by a slice coordinate
        if node in raw_to_ui_mapping:
            return TransformAction.PASS_THROUGH, raw_to_ui_mapping[node]
        return TransformAction.DEFER_TO_SLICE, None
        
    elif isinstance(node, Conv2dOutputCoordinate):
        return TransformAction.PASS_THROUGH, node
        
    elif isinstance(node, ReLUInputCoordinate):
        return TransformAction.PASS_THROUGH, node
        
    elif isinstance(node, ReLUOutputCoordinate):
        # Remove ReLU output nodes to simplify graph
        return TransformAction.REMOVE, None
        
    elif isinstance(node, ModelInputCoordinate):
        return TransformAction.PASS_THROUGH, node
        
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def _create_input_patch_node(
    slice_coord: Conv2dSliceCoordinate,
    raw_graph: nx.DiGraph, 
    raw_to_ui_mapping: Dict[Coordinate, UIGraphNode]
) -> Conv2dInputPatchNode:
    """Create a Conv2dInputPatchNode from a Conv2dSliceCoordinate and its parents."""
    
    # Get all Conv2dInputCoordinate parents of this slice coordinate
    input_coords = []
    for parent in raw_graph.predecessors(slice_coord):
        if isinstance(parent, Conv2dInputCoordinate):
            input_coords.append(parent)
    
    if not input_coords:
        raise ValueError(
            f"Conv2dSliceCoordinate {slice_coord} has no Conv2dInputCoordinate parents. "
            f"This violates the assumption that slice coordinates always have input coordinates as parents."
        )
    
    # Validate that all input coordinates are from the same layer and channel
    expected_layer = slice_coord.layer_name
    expected_channel = slice_coord.in_channel
    
    for coord in input_coords:
        if coord.layer_name != expected_layer:
            raise ValueError(
                f"Input coordinate {coord} has layer_name '{coord.layer_name}' but slice coordinate "
                f"{slice_coord} expects layer_name '{expected_layer}'. All input coordinates in a patch "
                f"must belong to the same layer."
            )
        if coord.channel != expected_channel:
            raise ValueError(
                f"Input coordinate {coord} has channel {coord.channel} but slice coordinate "
                f"{slice_coord} expects channel {expected_channel}. All input coordinates in a patch "
                f"must belong to the same channel."
            )
    
    # Calculate patch boundaries
    min_y = min(coord.y for coord in input_coords)
    max_y = max(coord.y for coord in input_coords) 
    min_x = min(coord.x for coord in input_coords)
    max_x = max(coord.x for coord in input_coords)
    
    # Create the patch node
    patch_node = Conv2dInputPatchNode(
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
    
    # Map all the input coordinates to this patch node
    for coord in input_coords:
        raw_to_ui_mapping[coord] = patch_node
        
    return patch_node


def _create_ui_edges(
    raw_graph: nx.DiGraph,
    ui_graph: nx.DiGraph, 
    raw_to_ui_mapping: Dict[Coordinate, UIGraphNode]
) -> None:
    """Create edges in the UI graph based on raw graph relationships."""
    
    processed_edges: Set[tuple[UIGraphNode, UIGraphNode]] = set()
    
    for raw_parent, raw_child in raw_graph.edges():
        # Skip edges where either node was removed from UI graph
        if raw_parent not in raw_to_ui_mapping or raw_child not in raw_to_ui_mapping:
            # Validate that missing nodes are expected to be missing
            if raw_parent not in raw_to_ui_mapping:
                # Check what action should have been taken for this node
                action, _ = _get_transform_action_and_node(raw_parent, raw_graph, {})
                if action not in (TransformAction.REMOVE, TransformAction.DEFER_TO_SLICE):
                    raise ValueError(
                        f"Raw graph node {raw_parent} was not mapped to any UI node, "
                        f"but its transform action is {action.value} which should result in a mapping."
                    )
                    
            if raw_child not in raw_to_ui_mapping:
                # Check what action should have been taken for this node
                action, _ = _get_transform_action_and_node(raw_child, raw_graph, {})
                if action not in (TransformAction.REMOVE, TransformAction.DEFER_TO_SLICE):
                    raise ValueError(
                        f"Raw graph node {raw_child} was not mapped to any UI node, "
                        f"but its transform action is {action.value} which should result in a mapping."
                    )
            continue
            
        ui_parent = raw_to_ui_mapping[raw_parent]
        ui_child = raw_to_ui_mapping[raw_child]
        
        # Avoid duplicate edges (important when multiple raw nodes map to same UI node)
        edge = (ui_parent, ui_child)  # Parent -> Child direction as in raw graph
        if edge not in processed_edges and ui_parent != ui_child:  # Avoid self-loops
            ui_graph.add_edge(ui_parent, ui_child)
            processed_edges.add(edge)