import { useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  Panel,
  type XYPosition,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getUIGraph } from '../fetchers/graph';
import { getLayoutedUIGraphNodes } from '../utils/graphLayout';
import { getUIGraphNodeId } from '../utils/uiGraphStrings';
import type { LinkNode, LinkEdge, UIGraphNodeLinkData } from '../fetchers/graph';
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
import { PruneGraphButton } from './PruneGraphButton';
import { PrunedGraphToggle } from './PrunedGraphToggle';
import { AttachedToSelectedNodeLayerSettings } from './prune_preview/AttachedToSelectedNodeTopKSumSliderPreview';
import { toggleDirection, type Direction } from '../types/direction';
import { makeEvenlySpacedLayout } from '../layouts';
import { getLayoutedLayerNodes } from '../layouts/layerLayout';

const CHILD_WIDTH = 130;
const CHILD_HEIGHT = 150;
const CHILD_PADDING = 10;
const CHILD_DIRECTION = "LR";
const PAGE_DIRECTION = "TB";
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
      throw new Error(`Coordinate string generation not implemented for ${JSON.stringify(coord)}`);
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
  fetcherType: FetcherType,
  position: XYPosition,
  parentId: string,
): Node {
  return {
    id: nodeId,
    type: 'ActivationFlowNode',
    height: CHILD_HEIGHT,
    width: CHILD_WIDTH,
    data: {
      handleDirection: null,
      coordinate: getCoordinateString(coord),
      modelAlias,
      inputAlias,
      workAlias,
      title: label,
      fetchers: DEFAULT_FETCHERS,
      fetcherType: fetcherType,
      overlayAlgorithm: overlay,
    },
    position,
    parentId,
    extent: 'parent',
  };
}

/**
 * Creates a default React Flow node.
 */
function createDefaultNode(nodeId: string, label: string, position: XYPosition, parentId: string): Node {
  return {
    id: nodeId,
    type: 'default',
    data: { label },
    position,
    parentId,
    extent: 'parent',
  };
}


const getLayerNode = (
  id: string, width: number, height: number, nodeCount: number, pageDirection: Direction
) => {
  return {
    id: id,
    type: 'LayerNode',
    position: { x: 0, y: 0 }, // empty position, filled by dagre
    width: width,
    height: height,
    data: {
      label: "",
      layerType: '',
      nodeCount: nodeCount,
      handleDirection: toggleDirection(pageDirection),
    },
  };
}


/**
 * Creates a React Flow node from a LinkNode, dispatching based on coordinate type.
 */
function createReactFlowNodes(
  nodeId: string,
  node: LinkNode,
  modelAlias: string,
  inputAlias: string,
  workAlias: string | undefined,
  fetcherType: FetcherType,
): [string, Node[]] {
  const uiNode = node.id;
  // const nodeId = getUIGraphNodeId(uiNode);
  const label = `${uiNode.layer_name} / ${uiNode.type}`;

  const nodes = [];
  const layerNodeId = `layer-${nodeId}`;
  const layout = makeEvenlySpacedLayout(1, CHILD_HEIGHT, CHILD_WIDTH, CHILD_PADDING, CHILD_DIRECTION);
  nodes.push(getLayerNode(layerNodeId, layout.parent.width, layout.parent.height, 1, CHILD_DIRECTION));

  const parentId = layerNodeId;
  const position = layout.children[0]
  switch (uiNode.type) {
    case 'Conv2dOutputCoordinate':
      const conv2dOutRect: OverlayAlgorithm = {
        type: 'DrawRect',
        start: [uiNode.x, uiNode.y],
        end: [uiNode.x + 1, uiNode.y + 1],
      };
      nodes.push(
        createActivationNode(
          nodeId, label, uiNode, modelAlias, inputAlias, workAlias, conv2dOutRect, fetcherType, position, parentId
        )
      );
      break;
    case 'SingleConv2dOpNode':
      const conv2dOpRect: OverlayAlgorithm = {
        type: 'DrawRect',
        start: [uiNode.input_patch.patch_min_x, uiNode.input_patch.patch_min_y],
        end: [uiNode.input_patch.patch_max_x + 1, uiNode.input_patch.patch_max_y + 1],
      };
      nodes.push(
        createActivationNode(
          nodeId, label, uiNode, modelAlias, inputAlias, workAlias, conv2dOpRect, fetcherType, position, parentId
        )
      );
      break;
    default:
      nodes.push(createDefaultNode(nodeId, label, position, parentId));
  }

  return [layerNodeId, nodes];
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
      return getLayoutedLayerNodes(nodes, edges, PAGE_DIRECTION);
    }
  });

  useEffect(() => {
    if (layoutedData) {
      console.log("setinggggggg", layoutedData.nodes);
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

