import { type ReactNode } from 'react';
import { ReactFlow, Background, Controls, Panel, Position, type ReactFlowProps, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSelectedNodeStore } from '../../stores/selectedNodeStore';
import { LayerNode } from '../nodes/LayerNode';
import { ActivationFlowNode } from '../nodes/ActivationFlowNode';
import { LayerSettings } from '../LayerSettings';
import dagre from '@dagrejs/dagre';
import type { Direction } from '../../types/direction';
import { DataTypeSelector } from './Controls/DataTypeSelector';
import { ColormapSelector } from './Controls/ColormapSelector';

const nodeTypes = {
  LayerNode,
  ActivationFlowNode,
};

interface SharedCanvasProps extends ReactFlowProps {
  children?: ReactNode;
  pageDirection?: Direction;
}

export default function SharedCanvas({ children, pageDirection, ...props }: SharedCanvasProps) {
  const { handleNodeClick, handlePaneClick } = useSelectedNodeStore();
  const { nodes, edges } = getLayoutedElements(props, pageDirection);
  const layoutedProps: ReactFlowProps = { ...props, nodes, edges }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow {...layoutedProps} nodeTypes={nodeTypes} onNodeClick={handleNodeClick} onPaneClick={handlePaneClick}>
        <Controls />
        <Background />

        <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
          <DataTypeSelector />
          <ColormapSelector />
          <LayerSettings />
        </Panel>

        {children}
      </ReactFlow>
    </div>
  );
}

const getLayoutedElements = (props: SharedCanvasProps, pageDirection?: Direction): { nodes: Node[], edges: Edge[] } => {
  const nodes = props.nodes;
  const edges = props.edges;
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
