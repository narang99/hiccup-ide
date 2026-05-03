# Node Layouting Documentation

## Overview

The application uses a two-tier layouting system: automatic layout for layers and manual positioning for activation nodes within layers.

## Layouting Architecture

### Tier 1: Dagre-js Automatic Layout (LayerNodes)

#### Purpose
- **Primary responsibility**: Layout relationships between neural network layers
- **Scope**: Only LayerNode components participate in dagre-js layout
- **Edges**: Only edges between LayerNodes are considered for layout calculation

#### Key Constraint
**Critical**: Only use LayerNode when creating relationships for dagre-js layout. This ensures clean, meaningful layer-to-layer connections without overwhelming the layout algorithm with individual activation relationships.

#### Benefits
- Automatic positioning based on layer dependencies
- Clean separation of concerns (layer structure vs content)
- Scalable to complex neural network architectures
- Prevents layout complexity from individual activations

### Tier 2: Manual Positioning (ActivationFlowNodes)

#### Purpose
- **Scope**: Individual activation nodes and saliency maps within LayerNodes
- **Positioning**: Absolute positioning relative to parent LayerNode
- **No edges**: ActivationFlowNode components have no edges between them

#### Implementation Location
- **Path**: `src/layouts/`
- **Main function**: `makeEvenlySpacedLayout()`

## Layout Functions

### Core Function: `makeEvenlySpacedLayout`

```typescript
makeEvenlySpacedLayout(
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
  direction: Direction,
): LayerGroupLayout
```

#### Parameters
- `numChannels`: Number of activation nodes to layout
- `childHeight`: Height of individual activation nodes
- `childWidth`: Width of individual activation nodes  
- `padding`: Space between nodes
- `direction`: "LR" (horizontal) or "TB" (vertical)

#### Returns: `LayerGroupLayout`
```typescript
interface LayerGroupLayout {
  children: XYPosition[];    // Positions for each activation node
  parent: { height: number, width: number };  // Required parent size
}
```

### Direction-Specific Layouts

#### Horizontal Layout (`makeEvenlySpacedHorizontalLayout`)
- **Layout**: Nodes arranged left-to-right
- **Width calculation**: `numChannels * childWidth + (numChannels + 1) * padding`
- **Height**: `padding + childHeight`
- **Position formula**: `x = channelIndex * childWidth + (channelIndex + 1) * padding`

#### Vertical Layout (`makeEvenlySpacedVerticalLayout`)
- **Layout**: Nodes arranged top-to-bottom
- **Height calculation**: `numChannels * childHeight + (numChannels + 1) * padding`
- **Width**: `padding + childWidth`
- **Position formula**: `y = channelIndex * childHeight + (channelIndex + 1) * padding`

## Standard Workflow

### Layer Creation Pattern

The standard workflow for creating complete layer nodes follows this pattern (example from `layerCreators/relu.ts`):

```typescript
export const createReLULayer = (
    modelNode: ModelNode,
    basePosition: { x: number; y: number },
    fetcherType: FetcherType,
    layerBlockHandleDirection: Direction,
    directionInsideLayerBlock: Direction = "LR",
    absMax?: number,
): Node[] => {
    const nodes: Node[] = [];
    
    // 1. Calculate layout
    const numChannels = modelNode.shape[1]; // Extract from model shape
    const childWidth = 130;
    const childHeight = 150;
    const padding = 10;
    const layout = makeEvenlySpacedLayout(
        numChannels, 
        childHeight, 
        childWidth, 
        padding, 
        directionInsideLayerBlock
    );

    // 2. Create parent LayerNode with calculated size
    nodes.push(makeParentLayerNode(
        modelNode.id, 
        basePosition, 
        layout.parent.width, 
        layout.parent.height, 
        numChannels, 
        layerBlockHandleDirection
    ));

    // 3. Create child ActivationFlowNodes at calculated positions
    for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
        const childPosition = layout.children[channelIndex];
        nodes.push(makeChannelNode(
            modelNode.id, 
            childPosition, 
            childHeight, 
            childWidth, 
            channelIndex
        ));
    }

    return nodes;
};
```

### Key Steps

1. **Extract Parameters**: Get number of channels from model data
2. **Calculate Layout**: Use `makeEvenlySpacedLayout()` to get positions and parent size
3. **Create Parent**: LayerNode with calculated width/height for dagre-js
4. **Create Children**: ActivationFlowNodes at calculated relative positions
5. **Set Relationships**: Children have `parentId` and `extent: 'parent'`

## Layout Constraints

### Parent-Child Positioning
- **Parent size**: Must accommodate all children plus padding
- **Child positioning**: Relative to parent LayerNode origin
- **Extent constraint**: Children use `extent: 'parent'` to stay within bounds

### Dagre-js Integration
- **Only LayerNodes**: Participate in automatic layout
- **Edge relationships**: Connect layers, not individual activations
- **Size awareness**: Parent LayerNode dimensions inform dagre-js spacing

### Future Extensibility
- **Current limitation**: Only evenly spaced layouts implemented
- **Potential additions**: Grid layouts, custom arrangements, adaptive spacing
- **Interface stability**: `LayerGroupLayout` interface supports various layout algorithms