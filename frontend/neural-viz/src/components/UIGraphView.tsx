import { useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getUIGraph } from '../fetchers/graph';
import { getLayoutedUIGraphNodes } from '../utils/graphLayout';
import { getUIGraphNodeText, getUIGraphNodeId } from '../utils/uiGraphStrings';
import type { LinkNode, LinkEdge } from '../fetchers/graph';
import { ActivationFlowNode } from './nodes/ActivationFlowNode';
import type { Coordinate, Conv2dOutputCoordinate } from '../types/coordinates';
import { DEFAULT_FETCHERS } from '../fetchers';

const nodeTypes = {
  ActivationNode: ActivationFlowNode,
};

/**
 * Returns the standardized coordinate string for activation fetching.
 * Only supports Conv2dOutputCoordinate currently.
 */
function getCoordinateString(coord: Coordinate): string {
  if (coord.type === 'Conv2dOutputCoordinate') {
    return `${coord.layer_name}.out_${coord.channel}`;
  }
  throw new Error(`Coordinate string generation not implemented for type: ${coord.type}`);
}

/**
 * Creates a React Flow node for a Conv2dOutputCoordinate.
 */
function createConv2dOutputNode(
  nodeId: string,
  label: string,
  coord: Conv2dOutputCoordinate,
  modelAlias: string,
  inputAlias: string,
  workAlias: string | undefined
): Node {
  return {
    id: nodeId,
    type: 'ActivationNode',
    height: 150,
    width: 130,
    data: {
      coordinate: getCoordinateString(coord),
      modelAlias,
      inputAlias,
      workAlias,
      title: label,
      fetchers: DEFAULT_FETCHERS,
      fetcherType: 'activation',
      overlayAlgorithm: {
        type: 'DrawRect',
        start: [coord.x, coord.y],
        end: [coord.x+1, coord.y+1],
      },
    },
    position: { x: 0, y: 0 },
  };
}

/**
 * Creates a default React Flow node.
 */
function createDefaultNode(nodeId: string, label: string): Node {
  return {
    id: nodeId,
    type: 'default',
    data: { label },
    position: { x: 0, y: 0 },
  };
}

/**
 * Creates a React Flow node from a LinkNode, dispatching based on coordinate type.
 */
function createReactFlowNode(
  node: LinkNode,
  modelAlias: string,
  inputAlias: string,
  workAlias: string | undefined
): Node {
  const uiNode = node.id;
  const nodeId = getUIGraphNodeId(uiNode);
  // const label = getUIGraphNodeText(uiNode);
  const label = `${uiNode.layer_name} / ${uiNode.type}`

  switch (uiNode.type) {
    case 'Conv2dOutputCoordinate':
      return createConv2dOutputNode(nodeId, label, uiNode, modelAlias, inputAlias, workAlias);
    default:
      return createDefaultNode(nodeId, label);
  }
}

const UIGraphView = () => {
  const { modelAlias, inputAlias, workAlias } = useParams();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // For now, we'll use a hardcoded coordinate to start the graph
  const startCoordinates = useMemo(() => [
    {
      type: "Conv2dOutputCoordinate",
      layer_name: "layers.2",
      layer_type: "conv2d",
      coordinate_type: "output",
      channel: 12,
      y: 2,
      x: 3,
    }
  ], []);

  const { data: layoutedData, isLoading, error } = useQuery({
    queryKey: ['uiGraph', modelAlias, inputAlias, workAlias, startCoordinates],
    queryFn: () => getUIGraph(modelAlias!, inputAlias!, workAlias!, startCoordinates),
    enabled: !!modelAlias && !!inputAlias && !!workAlias,
    select: (data) => {
      // Transform backend node-link data to React Flow format
      const rfNodes: Node[] = data.nodes.map((node: LinkNode) => 
        createReactFlowNode(node, modelAlias!, inputAlias!, workAlias)
      );

      const rfEdges: Edge[] = data.edges.map((edge: LinkEdge, index: number) => ({
        id: `e-${index}`,
        source: getUIGraphNodeId(edge.source),
        target: getUIGraphNodeId(edge.target),
      }));

      return getLayoutedUIGraphNodes(rfNodes, rfEdges, "TB");
    }
  });

  useEffect(() => {
    if (layoutedData) {
      setNodes(layoutedData.nodes);
      setEdges(layoutedData.edges);
    }
  }, [layoutedData, setNodes, setEdges]);

  if (isLoading) {
    return <div className="flex items-center justify-center h-full">Loading graph...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">Error: {(error as Error).message}</div>;
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default UIGraphView;
