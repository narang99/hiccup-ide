import type { Edge } from '@xyflow/react';
import { type ModelNode, type KernelExpandedState } from '../types/model';
import { createReactFlowEdge } from './edgeUtils';

export const createKernelEdges = (
  sourceNode: ModelNode, 
  targetNode: ModelNode
): Edge[] => {
  const edges: Edge[] = [];
  
  if (sourceNode.type === 'Conv2d' && targetNode.type === 'ReLU') {
    // Conv2d kernels to ReLU channels (1:1 mapping)
    const outChannels = sourceNode.params.out_channels as number;
    
    for (let i = 0; i < outChannels; i++) {
      const sourceKernelId = `${sourceNode.id}-kernel-${i}`;
      const targetChannelId = `${targetNode.id}-channel-${i}`;
      
      edges.push({
        id: `${sourceKernelId}-${targetChannelId}`,
        source: sourceKernelId,
        target: targetChannelId,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else if (sourceNode.type === 'ReLU' && targetNode.type === 'Conv2d') {
    // ReLU channels to Conv2d kernels (each channel connects to all kernels)
    const sourceChannels = sourceNode.shape.length > 1 ? sourceNode.shape[1] : 1;
    const targetKernels = targetNode.params.out_channels as number;
    
    for (let i = 0; i < sourceChannels; i++) {
      const sourceChannelId = `${sourceNode.id}-channel-${i}`;
      
      for (let j = 0; j < targetKernels; j++) {
        const targetKernelId = `${targetNode.id}-kernel-${j}`;
        
        edges.push({
          id: `${sourceChannelId}-${targetKernelId}`,
          source: sourceChannelId,
          target: targetKernelId,
          type: 'default',
          style: { stroke: '#6b7280', strokeWidth: 1, opacity: 0.7 },
        });
      }
    }
  } else if (sourceNode.type === 'Conv2d' && targetNode.type === 'Conv2d') {
    // Conv2d to Conv2d: connect kernels directly
    const outChannels = sourceNode.params.out_channels as number;
    const targetKernels = targetNode.params.out_channels as number;
    
    for (let i = 0; i < outChannels; i++) {
      const sourceKernelId = `${sourceNode.id}-kernel-${i}`;
      
      for (let j = 0; j < targetKernels; j++) {
        const targetKernelId = `${targetNode.id}-kernel-${j}`;
        
        edges.push({
          id: `${sourceKernelId}-${targetKernelId}`,
          source: sourceKernelId,
          target: targetKernelId,
          type: 'default',
          style: { stroke: '#6b7280', strokeWidth: 1, opacity: 0.7 },
        });
      }
    }
  } else if (sourceNode.type === 'Conv2d') {
    // Conv2d to other layer types: connect from kernels
    const outChannels = sourceNode.params.out_channels as number;
    
    for (let i = 0; i < outChannels; i++) {
      const sourceKernelId = `${sourceNode.id}-kernel-${i}`;
      
      edges.push({
        id: `${sourceKernelId}-${targetNode.id}`,
        source: sourceKernelId,
        target: targetNode.id,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else if (targetNode.type === 'Conv2d') {
    // Other layer types to Conv2d: connect to kernels
    const targetKernels = targetNode.params.out_channels as number;
    
    for (let j = 0; j < targetKernels; j++) {
      const targetKernelId = `${targetNode.id}-kernel-${j}`;
      
      edges.push({
        id: `${sourceNode.id}-${targetKernelId}`,
        source: sourceNode.id,
        target: targetKernelId,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else if (sourceNode.type === 'ReLU') {
    // ReLU channels to other layer types: connect all channels to target
    const sourceChannels = sourceNode.shape.length > 1 ? sourceNode.shape[1] : 1;
    
    for (let i = 0; i < sourceChannels; i++) {
      const sourceChannelId = `${sourceNode.id}-channel-${i}`;
      
      edges.push({
        id: `${sourceChannelId}-${targetNode.id}`,
        source: sourceChannelId,
        target: targetNode.id,
        type: 'default',
        style: { stroke: '#374151', strokeWidth: 2 },
      });
    }
  } else {
    // Non-Conv2d, non-ReLU connections: normal connection
    edges.push(createReactFlowEdge({ source: sourceNode.id, target: targetNode.id }));
  }
  
  return edges;
};

export const createKernelSliceEdges = (
  parentNode: ModelNode,
  kernelIndex: number,
  kernelExpandedState: KernelExpandedState
): Edge[] => {
  const edges: Edge[] = [];
  const kernelId = `${parentNode.id}-kernel-${kernelIndex}`;
  
  if (!kernelExpandedState[kernelId]) {
    return edges;
  }
  
  const inChannels = parentNode.params.in_channels as number;
  
  // Create edges for expanded kernel slices
  for (let i = 0; i < inChannels; i++) {
    const inputSliceId = `${parentNode.id}-kernel-${kernelIndex}-input-${i}`;
    const kernelSliceId = `${parentNode.id}-kernel-${kernelIndex}-slice-${i}`;
    const sliceOutputId = `${parentNode.id}-kernel-${kernelIndex}-output-${i}`;
    const sumNodeId = `${parentNode.id}-kernel-${kernelIndex}-sum`;
    
    // Input -> Kernel Slice
    edges.push({
      id: `${inputSliceId}-${kernelSliceId}`,
      source: inputSliceId,
      target: kernelSliceId,
      type: 'default',
      style: { stroke: '#059669', strokeWidth: 2 },
    });
    
    // Kernel Slice -> Slice Output
    edges.push({
      id: `${kernelSliceId}-${sliceOutputId}`,
      source: kernelSliceId,
      target: sliceOutputId,
      type: 'default',
      style: { stroke: '#7c3aed', strokeWidth: 2 },
    });
    
    // Slice Output -> Sum Node
    edges.push({
      id: `${sliceOutputId}-${sumNodeId}`,
      source: sliceOutputId,
      target: sumNodeId,
      type: 'default',
      style: { stroke: '#dc2626', strokeWidth: 2 },
    });
  }
  
  return edges;
};

export const createInputToKernelSliceEdges = (
  sourceNode: ModelNode,
  targetNode: ModelNode,
  kernelExpandedState: KernelExpandedState
): Edge[] => {
  const edges: Edge[] = [];
  
  if (targetNode.type !== 'Conv2d') {
    return edges;
  }
  
  const targetKernels = targetNode.params.out_channels as number;
  const inChannels = targetNode.params.in_channels as number;
  
  for (let kernelIndex = 0; kernelIndex < targetKernels; kernelIndex++) {
    const kernelId = `${targetNode.id}-kernel-${kernelIndex}`;
    
    if (kernelExpandedState[kernelId]) {
      // Connect source to expanded kernel input slices
      for (let i = 0; i < inChannels; i++) {
        const inputSliceId = `${targetNode.id}-kernel-${kernelIndex}-input-${i}`;
        
        edges.push({
          id: `${sourceNode.id}-${inputSliceId}`,
          source: sourceNode.id,
          target: inputSliceId,
          type: 'default',
          style: { stroke: '#374151', strokeWidth: 1, opacity: 0.8 },
        });
      }
    }
  }
  
  return edges;
};