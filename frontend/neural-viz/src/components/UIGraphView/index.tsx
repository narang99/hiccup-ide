import { useEffect, useMemo } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getUIGraph } from '../../fetchers/graph';
import { getUIGraphNodeId } from '../../utils/uiGraphStrings';
import type { LinkNode, LinkEdge, UIGraphNodeLinkData } from '../../fetchers/graph';
import { type FetcherType } from '../../fetchers';
import SharedCanvas from '../SharedCanvas';
import { DataTypeSelector } from '../SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from '../SharedCanvas/Controls/ColormapSelector';
import { useFetcherType } from '../../hooks/useFetcherType';
import { useGlobalStateControl } from '../../hooks/useGlobalStateControl';
import { PruneGraphButton } from '../PruneGraphButton';
import { PrunedGraphToggle } from '../PrunedGraphToggle';
import { AttachedToSelectedNodeLayerSettings } from '../prune_preview/AttachedToSelectedNodeTopKSumSliderPreview';
import { getLayoutedLayerNodes } from '../../layouts/layerLayout';
import { makeConv2dOpNodes, makeConv2dOutputCoordNodes, makeDefaultNodes } from './mkNode';
import { parseCoordinateFromURL } from '../../utils/urlCoordinateParser';

const CHILD_WIDTH = 130;
const CHILD_HEIGHT = 150;
const CHILD_PADDING = 10;
const CHILD_DIRECTION = "LR";
const PAGE_DIRECTION = "TB";

function createReactFlowNodes(
  nodeId: string,
  node: LinkNode,
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  fetcherType: FetcherType,
): [string, Node[]] {
  const uiNode = node.id;
  switch(uiNode.type) {
    case 'Conv2dOutputCoordinate':
      return makeConv2dOutputCoordNodes(
        nodeId, uiNode, modelAlias, inputAlias, workAlias, fetcherType, CHILD_HEIGHT, CHILD_WIDTH, CHILD_PADDING, CHILD_DIRECTION
      );
    case 'SingleConv2dOpNode':
      return makeConv2dOpNodes(
        nodeId, uiNode, modelAlias, inputAlias, workAlias, fetcherType, CHILD_HEIGHT, CHILD_WIDTH, CHILD_PADDING, CHILD_DIRECTION
      );
    default:
      return makeDefaultNodes(
        nodeId, uiNode, CHILD_HEIGHT, CHILD_WIDTH, CHILD_PADDING, CHILD_DIRECTION,
      )
  }
}

const convertToReactFlowGraph = (
  data: UIGraphNodeLinkData,
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  fetcherType: FetcherType,
): [Node[], Edge[]] => {
  // convert the nodes first
  const nodeIdByLayerNodeId = new Map<string, string>();
  const nodes: Node[] = []
  for (const node of data.nodes) {
    const nodeId = getUIGraphNodeId(node.id);
    const [layerNodeId, flowNodes] = createReactFlowNodes(
      nodeId, node, modelAlias, inputAlias, workAlias, fetcherType
    )
    nodeIdByLayerNodeId.set(nodeId, layerNodeId)
    nodes.push(...flowNodes);
  }
  // convert the edges
  const edges: Edge[] = data.edges.map((edge: LinkEdge, index: number) => {
    const sourceNodeId = getUIGraphNodeId(edge.source);
    const sourceLayerNodeId = nodeIdByLayerNodeId.get(sourceNodeId);
    const targetNodeId = getUIGraphNodeId(edge.target)
    const targetLayerNodeId = nodeIdByLayerNodeId.get(targetNodeId);

    if (sourceLayerNodeId === undefined || targetLayerNodeId === undefined) {
      throw Error(`could not find layer node id for either source or target, source=${sourceNodeId} target=${targetNodeId}`)
    }
    return ({
      id: `e-${index}`,
      source: sourceLayerNodeId,
      target: targetLayerNodeId,
    });
  });

  return [nodes, edges];
}

const UIGraphView = () => {
  const { modelAlias, inputAlias, workAlias } = useParams();
  const [searchParams] = useSearchParams();
  const { fetcherType } = useFetcherType();
  // const { isPruned } = usePruned();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useGlobalStateControl({
    nodes,
    fetcherType,
    setNodes,
  });

  const coordinateOrError = useMemo(() => parseCoordinateFromURL(searchParams), [searchParams]);

  const startCoordinates = useMemo(() => {
    if (typeof coordinateOrError === 'string') return [];
    return [coordinateOrError];
  }, [coordinateOrError]);

  const { data: layoutedData, isLoading, error } = useQuery({
    queryKey: ['uiGraph', modelAlias, inputAlias, workAlias, startCoordinates, fetcherType],
    queryFn: () => getUIGraph(modelAlias!, inputAlias!, workAlias!, startCoordinates),
    enabled: !!modelAlias && !!inputAlias && !!workAlias && typeof coordinateOrError !== 'string',
    select: (data) => {
      const [nodes, edges] = convertToReactFlowGraph(data, modelAlias!, inputAlias!, workAlias!, fetcherType)
      return getLayoutedLayerNodes(nodes, edges, PAGE_DIRECTION, 200);
    }
  });

  useEffect(() => {
    if (layoutedData) {
      setNodes(layoutedData.nodes);
      setEdges(layoutedData.edges);
    }
  }, [layoutedData, setNodes, setEdges]);

  if (typeof coordinateOrError === 'string') {
    return (
      <div className="p-8 flex flex-col items-center justify-center h-full text-center">
        <h2 className="text-xl font-bold text-red-600 mb-4">Invalid Graph Parameters</h2>
        <p className="bg-red-50 p-4 rounded border border-red-200 text-red-800 max-w-lg">
          {coordinateOrError}
        </p>
        <p className="mt-4 text-gray-600">
          Please provide valid coordinate parameters in the URL query string.
        </p>
      </div>
    );
  }

  if (isLoading) {
    return <div className="flex items-center justify-center h-full">Loading graph...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">Error: {(error as Error).message}</div>;
  }

  if (nodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8 bg-gray-900 text-white">
        <h2 className="text-xl font-bold mb-2">Graph is Empty</h2>
        <p className="text-gray-400">The backend returned no nodes for the specified coordinate.</p>
      </div>
    );
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

