import type { Edge } from '@xyflow/react';
import { type ModelNode, type ModelEdge, type ExpandedState } from '../types/model';

export const createReactFlowEdge = (modelEdge: ModelEdge): Edge => ({
  id: `${modelEdge.source}-${modelEdge.target}`,
  source: modelEdge.source,
  target: modelEdge.target,
  type: 'default',
  style: { stroke: '#374151', strokeWidth: 2 },
});

export const createChannelEdges = (
  sourceNode: ModelNode, 
  targetNode: ModelNode, 
  expandedState: ExpandedState
): Edge[] => {
  const edges: Edge[] = [];
  
  if (expandedState[sourceNode.id] && expandedState[targetNode.id]) {
    // Both nodes expanded: connect each output channel to each input channel
    const sourceChannels = sourceNode.params.out_channels as number;
    const targetChannels = targetNode.params.in_channels as number;
    
    for (let i = 0; i < sourceChannels; i++) {
      for (let j = 0; j < targetChannels; j++) {
        edges.push({
          id: `${sourceNode.id}-output-${i}-${targetNode.id}-input-${j}`,
          source: `${sourceNode.id}-output-${i}`,
          target: `${targetNode.id}-input-${j}`,
          type: 'default',
          style: { stroke: '#94a3b8', strokeWidth: 1, opacity: 0.6 },
        });
      }
    }
  } else if (expandedState[sourceNode.id]) {
    // Only source expanded: connect output channels to target node
    const sourceChannels = sourceNode.params.out_channels as number;
    
    for (let i = 0; i < sourceChannels; i++) {
      edges.push({
        id: `${sourceNode.id}-output-${i}-${targetNode.id}`,
        source: `${sourceNode.id}-output-${i}`,
        target: targetNode.id,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else if (expandedState[targetNode.id]) {
    // Only target expanded: connect source node to input channels
    const targetChannels = targetNode.params.in_channels as number;
    
    for (let j = 0; j < targetChannels; j++) {
      edges.push({
        id: `${sourceNode.id}-${targetNode.id}-input-${j}`,
        source: sourceNode.id,
        target: `${targetNode.id}-input-${j}`,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else {
    // Neither expanded: normal edge
    edges.push(createReactFlowEdge({ source: sourceNode.id, target: targetNode.id }));
  }
  
  return edges;
};