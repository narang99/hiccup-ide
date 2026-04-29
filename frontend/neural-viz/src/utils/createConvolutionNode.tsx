import React from 'react';
import type { Node } from '@xyflow/react';
import { type ModelNode, type ExpandedState } from '../types/model';
import { getNodeColor } from './nodeUtils';

export const createConvolutionNode = (
  modelNode: ModelNode, 
  index: number, 
  expandedState: ExpandedState, 
  toggleExpand: (nodeId: string) => void
): Node => {
  const isExpandable = modelNode.type === 'Conv2d' && modelNode.params.in_channels && modelNode.params.out_channels;
  const isExpanded = expandedState[modelNode.id];

  return {
    id: modelNode.id,
    type: 'default',
    position: { x: 300, y: index * 160 },
    data: {
      label: (
        <div className="text-center">
          <div className="flex items-center justify-center gap-2">
            <div className="font-bold text-sm">{modelNode.type}</div>
            {isExpandable && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleExpand(modelNode.id);
                }}
                className="text-xs bg-gray-200 hover:bg-gray-300 px-2 py-1 rounded"
              >
                {isExpanded ? '−' : '+'}
              </button>
            )}
          </div>
          <div className="text-xs text-gray-600">{modelNode.id}</div>
          {modelNode.shape.length > 0 && (
            <div className="text-xs text-gray-500">
              {modelNode.shape.join(' × ')}
            </div>
          )}
          {Object.keys(modelNode.params).length > 0 && (
            <div className="text-xs text-gray-500">
              {Object.entries(modelNode.params)
                .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(',') : value}`)
                .join(', ')}
            </div>
          )}
        </div>
      ),
    },
    style: {
      background: getNodeColor(modelNode.type),
      color: '#000',
      border: isExpanded ? '3px solid #059669' : '2px solid #000',
      borderRadius: '8px',
      padding: '10px',
      minWidth: '150px',
      zIndex: isExpanded ? 10 : 1,
    },
  };
};