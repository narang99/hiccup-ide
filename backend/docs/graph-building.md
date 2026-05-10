# Graph Building & Transformation Architecture

This document describes the architecture for building and transforming neural network dependency graphs in Hiccup IDE.

## Type-Driven Development

We follow a strict **Type-Driven Development** approach. The system is built around a single source of truth: the `Coordinate` discriminated union in `neural_data/types.py`. 

- **Single Union**: Both raw and UI graphs use the same `Coordinate` union.
- **Exhaustive Dispatch**: All transformation and traversal logic uses exhaustive pattern matching or type checking against this union.
- **Immutable Nodes**: Coordinates are immutable Pydantic models, serving as reliable unique identifiers in NetworkX graphs.

## 1. Raw Graph Construction

The **Raw Graph** represents the literal computational dependencies of the neural network.

- **Direction**: Edges flow from **Child → Parent** (following computational dependency).
- **Process**: Starting from a target coordinate, the system recursively identifies parent coordinates by inspecting the model's architecture. 
- **Filtering**: Pruning and saliency filters are applied during this phase, ensuring only relevant dependencies are included in the raw representation.

## 2. UI Graph Transformation (`RecurseStrategy`)

The **UI Graph** is a structural refinement of the Raw Graph, optimized for visualization. We use a **Chain of Responsibility** pattern to perform this transformation.

### The RecurseStrategy Contract

The transformation is governed by the `RecurseStrategy` protocol. Each strategy is a discrete unit of logic that evaluates a subgraph and decides how it should be represented in the UI.

#### Structural Decision making
When a strategy is invoked on a node, it must return one of two states:
1. **SKIP**: The strategy determines that this node (or the subgraph it roots) does not match its specific responsibility. It yields control to the next strategy in the chain.
2. **CONSUMED**: The strategy takes full responsibility for the node. It returns a list of coordinates (which may be the original node, new nodes, or an empty list) that represent the UI equivalent of that subgraph.

#### Recursive Delegation
Strategies are responsible for their own recursion. A strategy is passed a `main_strategy` callback, which it uses to process children. This allows strategies to:
- **Consolidate**: Gobble up a group of raw children and replace them with a single UI "Patch" node.
- **Omit**: Silently skip a node by returning the processed results of its children.
- **Pass-through**: Add the node to the UI graph and link it to the results of its recursive calls.

#### Caching & Integrity
- **Self-Managed Caching**: Each strategy is responsible for checking and updating the global transformation cache to prevent redundant processing.
- **Terminal Fallback**: A catch-all "Pass-Through" strategy ensures that every node in the raw graph is eventually accounted for if no specialized transformation applies.

## Summary of Flow

1. **Build Raw**: Generate a high-fidelity dependency graph from model architecture.
2. **Transform**: Pass the Raw Graph through the `RecurseStrategy` chain.
3. **Visualize**: Serialize the resulting UI-optimized `Coordinate` graph for the React Flow frontend.
