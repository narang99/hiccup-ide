import dagre from '@dagrejs/dagre';
import { Position, type Node, type Edge } from '@xyflow/react';
import type { Direction } from '../types/direction';

export const getLayoutedLayerNodes = (nodes: Node[], edges: Edge[], pageDirection?: Direction): { nodes: Node[], edges: Edge[] } => {
  const direction: Direction = (pageDirection === undefined) ? "LR" : pageDirection;

  if (nodes === undefined || edges === undefined) {
    return { nodes: [], edges: [] }
  }

  const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

  const isHorizontal = false;
  dagreGraph.setGraph({ rankdir: direction });

  // Only add LayerNode types to Dagre for positioning
  nodes.forEach((node) => {
    if (node.type === 'LayerNode') {
      dagreGraph.setNode(node.id, { width: node.width, height: node.height });
    }
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const newNodes = nodes.map((node) => {
    if (node.type === 'LayerNode') {
      const nodeWithPosition = dagreGraph.node(node.id);
      const width = node.width;
      const height = node.height;
      if (width === undefined || height === undefined) {
        throw Error(`invalid node ${node}: width and height cannot be undefined, width=${width} height=${height}`);
      }
      const actualWidth = typeof width === 'string' ? parseInt(width) : width;
      const actualHeight = typeof height === 'string' ? parseInt(height) : height;

      return {
        ...node,
        targetPosition: isHorizontal ? Position.Left : Position.Top,
        sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
        position: {
          x: nodeWithPosition.x - actualWidth / 2,
          y: nodeWithPosition.y - actualHeight / 2,
        },
      };
    } else {
      return node;
    }
  });

  return { nodes: newNodes, edges };
};
