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
  let nodeIdByLayerNodeId = new Map<string, string>();
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

