Note: you should read before reading this. And after reading this

Pre-requisite reads:
- `react-flow-architecture.md`  (same folder)
You should also read `node-layouting.md` as a sister document.  

## Layer Grouping Architecture

### Parent-Child Hierarchy

#### LayerNode (Parent)
- **Purpose**: Represents individual neural network layers
- **Type**: Parent nodes containing all layer-related visualizations
- **Edges**: Only LayerNode instances have edges between them
- **Layout**: Simplified edge structure makes dagre-js layout more efficient

#### ActivationFlowNode (Children)
- **Purpose**: Individual activations or saliency maps within a layer
- **Positioning**: Positioned relative to their parent LayerNode
- **DOM Structure**: Remain siblings in DOM despite logical parent-child relationship

### Node Management Patterns

#### Sibling Management
Since all nodes are DOM siblings (not true React parent-child):

1. **Filter by parentId**: Use parentId to identify child nodes
2. **Bulk updates**: Handle related nodes as groups during updates
3. **Prop-based updates**: Change data properties for custom nodes

```typescript
// Update all children of a specific layer
setNodes(prevNodes =>
  prevNodes.map(node => {
    if (node.parentId === layerId) {
      return { ...node, data: { ...node.data, newProperty: value } };
    }
    return node;
  })
);
```

#### Edge Simplification
- **Only LayerNode edges**: Connections exist only between layer parents
- **No activation edges**: Individual activations/saliency maps don't have direct edges
- **Layout benefits**: Reduces edge complexity for dagre-js automatic layout
- **Cleaner visualization**: Focuses on layer-to-layer data flow

### Component Types

| Component | Purpose | Parent | Edges | Children |
|-----------|---------|---------|--------|----------|
| LayerNode | Neural network layer | None | Yes | ActivationFlowNode |
| ActivationFlowNode | Individual activation/saliency | LayerNode | No | None |

### Benefits of This Architecture

1. **Performance**: Custom nodes with prop-based updates are more efficient
2. **Maintainability**: Clear separation between layer structure and content
3. **Layout**: Simplified edge structure works better with automatic layout algorithms
4. **Scalability**: Easy to add new visualization types as ActivationFlowNode children
5. **State Management**: Predictable update patterns for complex node hierarchies