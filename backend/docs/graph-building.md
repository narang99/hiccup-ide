# Graph Building Architecture

This document describes the graph building system for neural network coordinate dependencies in Hiccup IDE.

## Overview

The graph building system uses a **type-driven development** approach with discriminated unions to create dependency graphs of neural network coordinates. We maintain two distinct graph representations:

1. **Raw Graphs**: Computational dependency graphs used internally
2. **UI Graphs**: Visualization-optimized graphs consumed by the frontend

## Type-Driven Development Principles

All graph operations are built around discriminated union types that enable:
- **Type safety**: Each coordinate type has specific behaviors
- **Extensibility**: New coordinate types can be added cleanly  
- **Clear interfaces**: Functions dispatch based on coordinate type

### Core Type Pattern
```python
# Discriminated union with type field
Coordinate = Annotated[
    Union[Conv2dInputCoordinate, Conv2dOutputCoordinate, ...],
    Field(discriminator="type")
]

# Type-specific function dispatch
_COORDINATE_TO_PARENT_FUNC: dict[str, Callable] = {
    "Conv2dOutputCoordinate": get_parents_of_conv2d_output_coordinate,
    # ...
}
```

## Raw Graph Architecture

**Purpose**: Represent the computational dependencies between coordinates in the neural network.

**Location**: `neural_data/graph/graph_builder.py`

**Key Components**:
- `build_graph()`: Main entry point for raw graph construction
- `_COORDINATE_TO_PARENT_FUNC`: Type-to-function mapping for parent finding
- **Node Types**: Direct coordinate objects from `neural_data/types.py`
- **Edges**: Child → Parent relationships (computational flow direction)

**Example Raw Graph Flow**:
```
Conv2dOutputCoordinate 
    ↓
Conv2dSliceCoordinate 
    ↓
Conv2dInputCoordinate (multiple for receptive field)
    ↓  
ReLUOutputCoordinate (from previous layer)
```

## UI Graph Architecture  

**Purpose**: Transform raw graphs into visualization-friendly representations.

**Location**: `neural_data/graph/ui_graph_builder.py` + `ui_graph_transformer.py`

**Key Transformations**:

### Conv2d Receptive Field Consolidation
- **Raw**: `Conv2dSliceCoordinate → [Conv2dInputCoordinate]` (many nodes)
- **UI**: `Conv2dInputPatchNode` (single node with patch info)

### Pass-Through Types
These coordinate types remain unchanged in UI graphs:
- `Conv2dOutputCoordinate` 
- `ReLUInputCoordinate`
- `ReLUOutputCoordinate`
- `ModelInputCoordinate`

### Excluded Types
These are transformed away in UI graphs:
- `Conv2dSliceCoordinate` → becomes part of `Conv2dInputPatchNode`
- `Conv2dInputCoordinate` → consolidated into `Conv2dInputPatchNode`

## NetworkX Integration

**Why NetworkX**: 
- Efficient graph traversal and manipulation
- Supports heterogeneous node types (different coordinate objects)
- Rich ecosystem of graph algorithms
- Good performance for our graph sizes

**Node Identity**: 
- Raw graphs use coordinate objects directly as node IDs (they're immutable)
- UI graphs use transformed coordinate objects as node IDs
- Edge direction: Child → Parent (following computational dependencies)

## Usage Patterns

### Building Raw Graphs
```python
from neural_data.graph.graph_builder import build_graph

# Build from any starting coordinate
raw_graph = build_graph(start_coordinate, model_dfn, filter_func)

# Graph contains actual coordinate objects as nodes
for node in raw_graph.nodes():
    print(f"Node type: {node.type}")
    print(f"Parents: {list(raw_graph.successors(node))}")
```

### Building UI Graphs  
```python
from neural_data.graph.ui_graph_builder import build_ui_graph

# Build UI-optimized graph
ui_graph = build_ui_graph(start_coordinate, model_dfn, filter_func)

# Graph contains UI node types
for node in ui_graph.nodes():
    if node.type == "Conv2dInputPatchNode":
        print(f"Patch: ({node.patch_min_y},{node.patch_min_x}) to ({node.patch_max_y},{node.patch_max_x})")
        print(f"Input coordinates: {len(node.input_coordinates)}")
```

## Adding New Coordinate Types

### For Raw Graphs
1. Add new coordinate type to `neural_data/types.py`
2. Add parent-finding function to `neural_data/graph/parent_coordinates/`
3. Register in `_COORDINATE_TO_PARENT_FUNC` mapping

### For UI Graphs  
1. Decide: pass-through or transform?
2. **Pass-through**: Add to `UIGraphNode` union in `ui_graph_types.py`
3. **Transform**: Create new UI node type + transformation logic in `ui_graph_transformer.py`

## Testing Strategy

- **Raw Graphs**: Test coordinate parent-finding functions individually
- **UI Graphs**: Test transformation logic and relationship preservation  
- **Integration**: Test end-to-end graph building from real model definitions

## Performance Considerations

- **Immutable Nodes**: Coordinate objects are immutable (safe for hashing/caching)
- **Lazy Evaluation**: Consider for very large graphs
- **Memory**: UI graphs typically smaller than raw (due to consolidation)
- **Filtering**: Applied at raw graph level before transformation (more efficient)
