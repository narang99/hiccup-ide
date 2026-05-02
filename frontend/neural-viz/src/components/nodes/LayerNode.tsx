import { type NodeProps, type Node } from '@xyflow/react';
import SingleOrNoHandle from '../SingleOrNoHandle';
import type { Direction } from '../../types/direction';

export interface LayerNodeData extends Record<string, unknown> {
  label: string;
  layerType: 'Conv2d' | 'ReLU' | 'Linear' | 'Flatten' | 'Input' | 'Output';
  nodeCount?: number;
  handleDirection: Direction | null;
}

const getLayerColor = (layerType: string) => {
  switch (layerType) {
    case 'Conv2d':
      return {
        background: 'rgba(96, 165, 250, 0.1)',
        border: '2px solid rgba(96, 165, 250, 0.3)',
        accent: '#60a5fa'
      };
    case 'ReLU':
      return {
        background: 'rgba(251, 191, 36, 0.1)',
        border: '2px solid rgba(251, 191, 36, 0.3)',
        accent: '#fbbf24'
      };
    case 'Linear':
      return {
        background: 'rgba(167, 139, 250, 0.1)',
        border: '2px solid rgba(167, 139, 250, 0.3)',
        accent: '#a78bfa'
      };
    case 'Flatten':
      return {
        background: 'rgba(244, 114, 182, 0.1)',
        border: '2px solid rgba(244, 114, 182, 0.3)',
        accent: '#f472b6'
      };
    case 'Input':
      return {
        background: 'rgba(74, 222, 128, 0.1)',
        border: '2px solid rgba(74, 222, 128, 0.3)',
        accent: '#4ade80'
      };
    case 'Output':
      return {
        background: 'rgba(248, 113, 113, 0.1)',
        border: '2px solid rgba(248, 113, 113, 0.3)',
        accent: '#f87171'
      };
    default:
      return {
        background: 'rgba(156, 163, 175, 0.1)',
        border: '2px solid rgba(156, 163, 175, 0.3)',
        accent: '#9ca3af'
      };
  }
};

export type LayerNodeType = Node<LayerNodeData, 'LayerNode'>;

export const LayerNode = ({ data, selected, width, height }: NodeProps) => {
  const typedData = data as LayerNodeData;

  const colors = getLayerColor(typedData.layerType);

  return (
    <div
      style={{
        background: colors.background,
        border: selected ? `2px solid ${colors.accent}` : colors.border,
        borderRadius: '8px',
        padding: '12px',
        width: width || '120px',
        height: height || '100px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        backdropFilter: 'blur(10px)',
        boxShadow: selected ? `0 0 20px ${colors.accent}40` : '0 2px 10px rgba(0,0,0,0.1)',
        transition: 'all 0.2s ease',
      }}
    >
      <SingleOrNoHandle handleDirection={typedData.handleDirection} badgeColor={colors.accent} />

      <div style={{
        fontSize: '12px',
        fontWeight: 700,
        color: colors.accent,
        textAlign: 'center',
        marginBottom: '4px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        {typedData.layerType}
      </div>

      {typedData.nodeCount && (
        <div style={{
          fontSize: '10px',
          color: 'rgba(255,255,255,0.7)',
          textAlign: 'center',
        }}>
          {typedData.nodeCount} {typedData.layerType === 'Conv2d' ? 'kernels' : 'channels'}
        </div>
      )}

    </div>
  );
};