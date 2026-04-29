import React from 'react';
import type { Node } from '@xyflow/react';
import { type ModelNode } from '../types/model';

export const getNodeColor = (nodeType: string): string => {
  switch (nodeType) {
    case 'Input':
      return '#4ade80';
    case 'Output':
      return '#f87171';
    case 'Conv2d':
      return '#60a5fa';
    case 'Conv2d-Channel':
      return '#93c5fd';
    case 'Linear':
      return '#a78bfa';
    case 'ReLU':
      return '#fbbf24';
    case 'Flatten':
      return '#f472b6';
    default:
      return '#9ca3af';
  }
};

export const createChannelNode = (
  parentNode: ModelNode, 
  channelIndex: number, 
  isInput: boolean, 
  expandedNodePosition: { x: number; y: number }
): Node => {
  const channelType = isInput ? 'input' : 'output';
  const nodeId = `${parentNode.id}-${channelType}-${channelIndex}`;
  const offsetX = isInput ? -200 : 200;
  const offsetY = (channelIndex - (isInput ? (parentNode.params.in_channels as number) : (parentNode.params.out_channels as number)) / 2) * 40;

  return {
    id: nodeId,
    type: 'default',
    position: { 
      x: expandedNodePosition.x + offsetX, 
      y: expandedNodePosition.y + offsetY 
    },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-xs">{channelType.toUpperCase()}</div>
          <div className="text-xs text-gray-600">Ch {channelIndex}</div>
        </div>
      ),
    },
    style: {
      background: getNodeColor('Conv2d-Channel'),
      color: '#000',
      border: '1px solid #374151',
      borderRadius: '6px',
      padding: '8px',
      minWidth: '60px',
      fontSize: '10px',
    },
    // parentNode: parentNode.id,
  };
};