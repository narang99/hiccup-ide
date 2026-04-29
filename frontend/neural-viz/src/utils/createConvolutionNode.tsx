import React from 'react';
import type { Node } from '@xyflow/react';
import { type ModelNode } from '../types/model';
import { getNodeColor } from './nodeUtils';

export const createConvolutionNode = (
  modelNode: ModelNode, 
  index: number
): Node => {
  // Conv2d layers now always show in collapsed mode (kernels shown separately)
  return {
    id: modelNode.id,
    type: 'default',
    position: { x: 300, y: index * 200 },
    data: {
      label: (
        <div className="text-center">
          <div className="font-bold text-sm">{modelNode.type}</div>
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
      border: '2px solid #000',
      borderRadius: '8px',
      padding: '10px',
      minWidth: '150px',
      zIndex: 1,
    },
  };
};