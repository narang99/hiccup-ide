import type { Node } from '@xyflow/react';
import { type ModelNode } from '../types/model';
import { type NodeFetchers, type FetcherType } from '../fetchers';
import BaseActivationNode from '../components/nodes/BaseActivationNode';
import { Link } from 'react-router-dom';
import { type HandleDirection } from '../components/nodes/ActivationFlowNode';

export const getKernelNodeColor = (nodeType: string): string => {
  switch (nodeType) {
    case 'Output-Kernel':
      return '#3b82f6'; // Blue for output kernels
    case 'Input-Slice':
      return '#10b981'; // Green for input slices
    case 'Kernel-Slice':
      return '#8b5cf6'; // Purple for kernel slices
    case 'Slice-Output':
      return '#f59e0b'; // Amber for slice outputs
    case 'Sum-Node':
      return '#ef4444'; // Red for summation
    default:
      return '#6b7280';
  }
};

// Create output kernel node (default expanded view)
export const createOutputKernelNode = (
  parentNode: ModelNode,
  kernelIndex: number,
  height: number,
  width: number,
  position: { x: number; y: number },
  fetchers?: NodeFetchers,
  fetcherType?: FetcherType,
  handleDirection: HandleDirection = null,
): Node => {
  const kernelId = `${parentNode.id}-kernel-${kernelIndex}`;
  const coordinate = `${parentNode.id}.out_${kernelIndex}`;

  const ks = parentNode.params.kernel_size as number[] | undefined;
  const title = ks ? `${ks[0]}×${ks[1]}` : parentNode.id;

  return {
    id: kernelId,
    type: 'ActivationFlowNode',
    width: width,
    height: height,
    position: position,
    data: {
      coordinate,
      fetchers,
      fetcherType,
      title,
      badgeLabel: kernelIndex !== undefined ? `K${kernelIndex}` : undefined,
      badgeColor: "#60a5fa",
      handleDirection,
      link: `/kernel/${parentNode.id}/${kernelIndex}`,
    },
    style: {
      background: 'transparent',
      border: '1px solid rgba(96, 165, 250, 0.35)',
      borderRadius: '6px',
      padding: 0,
      fontSize: '10px',
      overflow: 'hidden',
    },
    parentId: parentNode.id,
    extent: 'parent',
  };
};

// Create input channel slice node (when kernel is expanded)
export const createInputSliceNode = (
  parentNode: ModelNode,
  kernelIndex: number,
  inputChannelIndex: number,
  basePosition: { x: number; y: number }
): Node => {
  const sliceId = `${parentNode.id}-kernel-${kernelIndex}-input-${inputChannelIndex}`;

  return {
    id: sliceId,
    type: 'default',
    position: {
      x: basePosition.x - 150,
      y: basePosition.y + (inputChannelIndex - (parentNode.params.in_channels as number) / 2) * 40
    },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-xs">IN{inputChannelIndex}</div>
          <div className="text-xs text-gray-500">Input</div>
        </div>
      ),
    },
    style: {
      background: getKernelNodeColor('Input-Slice'),
      color: '#fff',
      border: '1px solid #374151',
      borderRadius: '4px',
      padding: '6px',
      minWidth: '40px',
      fontSize: '9px',
    },
  };
};

// Create kernel slice node (computation element)
export const createKernelSliceNode = (
  parentNode: ModelNode,
  kernelIndex: number,
  inputChannelIndex: number,
  basePosition: { x: number; y: number }
): Node => {
  const sliceId = `${parentNode.id}-kernel-${kernelIndex}-slice-${inputChannelIndex}`;

  return {
    id: sliceId,
    type: 'default',
    position: {
      x: basePosition.x,
      y: basePosition.y + (inputChannelIndex - (parentNode.params.in_channels as number) / 2) * 40
    },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-xs">K{kernelIndex}:{inputChannelIndex}</div>
          <div className="text-xs text-gray-500">Kernel</div>
        </div>
      ),
    },
    style: {
      background: getKernelNodeColor('Kernel-Slice'),
      color: '#fff',
      border: '1px solid #374151',
      borderRadius: '4px',
      padding: '6px',
      minWidth: '45px',
      fontSize: '9px',
    },
  };
};

// Create slice output node (after convolution)
export const createSliceOutputNode = (
  parentNode: ModelNode,
  kernelIndex: number,
  inputChannelIndex: number,
  basePosition: { x: number; y: number }
): Node => {
  const outputId = `${parentNode.id}-kernel-${kernelIndex}-output-${inputChannelIndex}`;

  return {
    id: outputId,
    type: 'default',
    position: {
      x: basePosition.x + 150,
      y: basePosition.y + (inputChannelIndex - (parentNode.params.in_channels as number) / 2) * 40
    },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-xs">OUT{inputChannelIndex}</div>
          <div className="text-xs text-gray-500">Slice</div>
        </div>
      ),
    },
    style: {
      background: getKernelNodeColor('Slice-Output'),
      color: '#fff',
      border: '1px solid #374151',
      borderRadius: '4px',
      padding: '6px',
      minWidth: '40px',
      fontSize: '9px',
    },
  };
};

// Create summation node (combines all slice outputs)
export const createSumNode = (
  parentNode: ModelNode,
  kernelIndex: number,
  basePosition: { x: number; y: number }
): Node => {
  const sumId = `${parentNode.id}-kernel-${kernelIndex}-sum`;

  return {
    id: sumId,
    type: 'default',
    position: {
      x: basePosition.x + 300,
      y: basePosition.y
    },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-sm">Σ</div>
          <div className="text-xs text-gray-500">Sum</div>
        </div>
      ),
    },
    style: {
      background: getKernelNodeColor('Sum-Node'),
      color: '#fff',
      border: '2px solid #374151',
      borderRadius: '6px',
      padding: '8px',
      minWidth: '40px',
      fontSize: '10px',
    },
  };
};