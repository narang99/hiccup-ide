import { useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getUIGraph } from '../fetchers/graph';
import { getLayoutedUIGraphNodes } from '../utils/graphLayout';
import { getUIGraphNodeId } from '../utils/uiGraphStrings';
import type { LinkNode, LinkEdge } from '../fetchers/graph';
import type {
  Coordinate
} from '../types/coordinates';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import type { OverlayAlgorithm } from '../types/overlay';
import SharedCanvas from './SharedCanvas';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { useFetcherType } from '../hooks/useFetcherType';
import { useGlobalStateControl } from '../hooks/useGlobalStateControl';
import { usePruned } from '../hooks/usePruned';
import { PruneGraphButton } from './PruneGraphButton';
import { PrunedGraphToggle } from './PrunedGraphToggle';
import { AttachedToSelectedNodeLayerSettings } from './prune_preview/AttachedToSelectedNodeTopKSumSliderPreview';

/**
 * Returns the standardized coordinate string for activation fetching.
 */
function getCoordinateString(coord: Coordinate): string {
  switch (coord.type) {
    case 'Conv2dOutputCoordinate':
      return `${coord.layer_name}.out_${coord.channel}`
    case 'SingleConv2dOpNode':
      return `${coord.input_patch.layer_name}.out_${coord.input_patch.channel}`
    default:
      // Fallback for types that might not have channel explicitly but are still activation-like
      if ('layer_name' in coord) {
          return `${coord.layer_name}.out_0`;
      }
      throw new Error(`Coordinate string generation not implemented for type: ${coord.type}`);
  }
}

/**
 * Creates a React Flow node for an ActivationFlowNode.
 */
function createActivationNode(
  nodeId: string,
  label: string,
  coord: Coordinate,
  modelAlias: string,
  inputAlias: string,
  workAlias: string | undefined,
  overlay: OverlayAlgorithm,
  fetcherType: FetcherType
): Node {
  return {
    id: nodeId,
    type: 'ActivationFlowNode',
    height: 150,
    width: 130,
    data: {
      coordinate: getCoordinateString(coord),
      modelAlias,
      inputAlias,
      workAlias,
      title: label,
      fetchers: DEFAULT_FETCHERS,
      fetcherType: fetcherType,
      overlayAlgorithm: overlay,
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
  workAlias: string | undefined,
  fetcherType: FetcherType
): Node {
  const uiNode = node.id;
  const nodeId = getUIGraphNodeId(uiNode);
  const label = `${uiNode.layer_name} / ${uiNode.type}`;

  switch (uiNode.type) {
    case 'Conv2dOutputCoordinate':
      return createActivationNode(nodeId, label, uiNode, modelAlias, inputAlias, workAlias, {
        type: 'DrawRect',
        start: [uiNode.x, uiNode.y],
        end: [uiNode.x + 1, uiNode.y + 1],
      }, fetcherType);
    case 'SingleConv2dOpNode':
      return createActivationNode(nodeId, label, uiNode, modelAlias, inputAlias, workAlias, {
        type: 'DrawRect',
        start: [uiNode.input_patch.patch_min_x, uiNode.input_patch.patch_min_y],
        end: [uiNode.input_patch.patch_max_x + 1, uiNode.input_patch.patch_max_y + 1],
      }, fetcherType);
    default:
      return createDefaultNode(nodeId, label);
  }
}

const UIGraphView = () => {
  const { modelAlias, inputAlias, workAlias } = useParams();
  const { fetcherType } = useFetcherType();
  // const { isPruned } = usePruned();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useGlobalStateControl({
    nodes,
    fetcherType,
    setNodes,
  });

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
    queryKey: ['uiGraph', modelAlias, inputAlias, workAlias, startCoordinates, fetcherType],
    queryFn: () => getUIGraph(modelAlias!, inputAlias!, workAlias!, startCoordinates),
    enabled: !!modelAlias && !!inputAlias && !!workAlias,
    select: (data) => {
      // Transform backend node-link data to React Flow format
      const rfNodes: Node[] = data.nodes.map((node: LinkNode) => 
        createReactFlowNode(node, modelAlias!, inputAlias!, workAlias, fetcherType)
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
    <SharedCanvas
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
    >
        <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
            <DataTypeSelector />
            <ColormapSelector />
            <AttachedToSelectedNodeLayerSettings />
            <PrunedGraphToggle />
            <PruneGraphButton />
        </Panel>
    </SharedCanvas>
  );
};

export default UIGraphView;

