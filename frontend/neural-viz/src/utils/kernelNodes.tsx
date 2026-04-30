import type { Node } from '@xyflow/react';
import { type ModelNode } from '../types/model';
import { type NodeFetchers } from '../fetchers';
import { ActivationDisplay } from '../components/ActivationDisplay';

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
  basePosition: { x: number; y: number },
  fetchers?: NodeFetchers
): Node => {
  const kernelId = `${parentNode.id}-kernel-${kernelIndex}`;
  const coordinate = `${parentNode.id}.out_${kernelIndex}`;
  
  const handleKernelClick = () => {
    // Navigate to kernel detail view
    window.location.href = `/kernel/${parentNode.id}/${kernelIndex}`;
  };
  
  return {
    id: kernelId,
    type: 'default',
    position: {
      x: basePosition.x + (kernelIndex - (parentNode.params.out_channels as number) / 2) * 120, // 120px spacing for 50px nodes + 70px gap
      y: basePosition.y // Same vertical level as parent layer
    },
    data: {
      label: (
        <div 
          className="text-center cursor-pointer hover:opacity-80"
          onClick={handleKernelClick}
        >
          <div className="flex flex-col items-center gap-1">
            <div className="font-bold text-xs">K{kernelIndex}</div>
            
            {/* Activation Display */}
            {fetchers?.activation ? (
              <ActivationDisplay 
                coordinate={coordinate}
                fetcher={fetchers.activation}
                maxSize={40}
              />
            ) : (
              <div className="w-10 h-10 bg-gray-400 rounded border" />
            )}
            
            <div className="text-xs text-gray-200">
              {parentNode.params.kernel_size ? 
                `${(parentNode.params.kernel_size as number[])[0]}×${(parentNode.params.kernel_size as number[])[1]}` : 
                'Conv'}
            </div>
            <div className="text-xs text-gray-300">🔍</div>
          </div>
        </div>
      ),
    },
    style: {
      background: getKernelNodeColor('Output-Kernel'),
      color: '#fff',
      border: '1px solid #374151',
      borderRadius: '6px',
      padding: '8px',
      minWidth: '80px',
      fontSize: '10px',
    },
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