# React Flow Architecture Documentation

## Overview

This application uses React Flow to build interactive graphs visualizing the internals of convolutional neural networks (CNNs). The primary focus is analyzing convolutional backbone layers, with placeholder representations for linear head layers.

## React Flow Data Model

### Core Concepts

React Flow differs from standard React in how it manages component lifecycles and state updates:

- **Standard React**: Components re-render via DOM diffing when props/state change
- **React Flow**: All nodes are siblings with lifecycles managed by the library itself

### Node Creation and Updates

#### Initial Setup
- Create nodes initially and pass them to the ReactFlow canvas
- ReactFlow renders and manages these nodes independently

#### Updating Nodes
- Use `setNodes()` to update existing nodes
- **Critical**: Only return references to updated nodes in setNodes
- React Flow uses these references to determine which nodes need updates

#### State Management Pattern
```typescript
// ✅ Correct - only return updated node references
setNodes(prevNodes => 
  prevNodes.map(node => 
    node.id === targetId 
      ? { ...node, data: { ...node.data, newProp: value } }
      : node
  )
);
```

### Custom Nodes vs Data Components

#### Problem with Data Components
- Passing React components in `data` field works initially
- **Issue**: Updates require rebuilding entire component with all initial props
- Makes selective property updates complex and inefficient

#### Custom Nodes Solution
- Custom nodes are React components with lifecycle fully managed by React Flow
- Pass **props** to `data` instead of JSX components
- Enables easy updates by changing only specific properties in data

```typescript
// ✅ Custom node pattern
const customNode = {
  id: 'node-1',
  type: 'customNodeType',
  data: {
    prop1: 'value1',
    prop2: 'value2'
  }
};

// Easy updates
setNodes(prevNodes => 
  prevNodes.map(node => 
    node.id === 'node-1'
      ? { ...node, data: { ...node.data, prop1: 'newValue' } }
      : node
  )
);
```
