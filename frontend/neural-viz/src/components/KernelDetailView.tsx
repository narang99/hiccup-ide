import { useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useFetcherType } from '../hooks/useFetcherType';
import { useModelData } from '../hooks/useModelData';
import SharedCanvas from './SharedCanvas';
import { type HandleDirection } from './nodes/ActivationFlowNode';
import { makeEvenlySpacedLayout } from '../layouts';
import type { Direction } from '../types/direction';

const getNodeShowingActivation = (
  id: string, 
  position: { x: number, y: number }, 
  title: string, 
  coordinate: string, 
  fetcherType: FetcherType = "activation", 
  parentId?: string, 
  width?: number, 
  height?: number,
  handleDirection: HandleDirection = null,
): Node => {
  return ({
    id: id,
    type: 'ActivationFlowNode',
    position: position,
    data: {
      coordinate: coordinate,
      fetchers: DEFAULT_FETCHERS,
      fetcherType: fetcherType,
      maxSize: 84,
      title: title,
      handleDirection,
    },
    width: width,
    height: height,
    style: {
      background: 'transparent',
      border: '1px solid rgba(245, 158, 11, 0.35)',
      borderRadius: '6px',
      padding: 0,
      fontSize: '10px',
      overflow: 'hidden',
    },
    parentId: parentId,
    extent: parentId ? 'parent' : undefined,
  });
};

interface LayoutConfig {
  childWidth: number;
  childHeight: number;
  padding: number;
  direction: Direction;
}

const createChannelSlice = (
  i: number,
  nodeId: string,
  kernelIdx: number,
  fetcherType: FetcherType,
  config: LayoutConfig,
): Node[] => {
  const sliceLayerId = `slice-${i}`;
  const sliceLayout = makeEvenlySpacedLayout(3, config.childHeight, config.childWidth, config.padding, config.direction);

  return [
    {
      id: sliceLayerId,
      type: 'LayerNode',
      position: { x: 0, y: 0 },
      width: sliceLayout.parent.width,
      height: sliceLayout.parent.height,
      data: {
        label: `Channel ${i} Slice`,
        layerType: 'Conv2d',
        nodeCount: 3,
        handleDirection: config.direction,
      },
    },
    getNodeShowingActivation(
      `input-${i}`,
      sliceLayout.children[0],
      "Input",
      `${nodeId}.out_${kernelIdx}.in_${i}.input`,
      fetcherType,
      sliceLayerId,
      config.childWidth,
      config.childHeight,
    ),
    getNodeShowingActivation(
      `kernel-slice-${i}`,
      sliceLayout.children[1],
      `K${kernelIdx}:${i}`,
      `${nodeId}.out_${kernelIdx}.in_${i}`,
      'weight',
      sliceLayerId,
      config.childWidth,
      config.childHeight,
    ),
    getNodeShowingActivation(
      `output-${i}`,
      sliceLayout.children[2],
      "Output",
      `${nodeId}.out_${kernelIdx}.in_${i}`,
      fetcherType,
      sliceLayerId,
      config.childWidth,
      config.childHeight,
    ),
  ];
};

const createSumGroup = (
  nodeId: string,
  kernelIdx: number,
  fetcherType: FetcherType,
  config: LayoutConfig,
): Node[] => {
  const sumLayerId = 'sum-layer';
  const sumLayout = makeEvenlySpacedLayout(1, config.childHeight, config.childWidth, config.padding, config.direction);

  return [
    {
      id: sumLayerId,
      type: 'LayerNode',
      position: { x: 0, y: 0 },
      width: sumLayout.parent.width,
      height: sumLayout.parent.height,
      data: {
        label: 'Sum',
        layerType: 'Output',
        nodeCount: 1,
        handleDirection: config.direction,
      },
    },
    getNodeShowingActivation(
      'sum',
      sumLayout.children[0],
      "Sum",
      `${nodeId}.out_${kernelIdx}`,
      fetcherType,
      sumLayerId,
      config.childWidth,
      config.childHeight
    ),
  ];
};

const createSliceToSumEdges = (inChannels: number, sumLayerId: string): Edge[] => {
  const edges: Edge[] = [];
  for (let i = 0; i < inChannels; i++) {
    edges.push({
      id: `slice-${i}-to-sum`,
      source: `slice-${i}`,
      target: sumLayerId,
      type: 'default',
      style: { stroke: '#dc2626', strokeWidth: 2 },
    });
  }
  return edges;
};

export default function KernelDetailView() {
  const { fetcherType } = useFetcherType();
  const pageDirection: Direction = "LR";
  const { nodeId, kernelIndex } = useParams<{ nodeId: string; kernelIndex: string }>();
  const navigate = useNavigate();
  const { modelData } = useModelData("example-model");

  const generateKernelDetailView = useCallback((data: ModelData, nodeId: string, kernelIdx: number): { nodes: Node[], edges: Edge[] } | null => {
    const targetNode = data.nodes.find(n => n.id === nodeId);
    if (!targetNode || targetNode.type !== 'Conv2d') return null;

    const inChannels = targetNode.params.in_channels as number;
    let nodes: Node[] = [];

    const sliceConfig: LayoutConfig = {
      childWidth: 130,
      childHeight: 150,
      padding: 10,
      direction: pageDirection,
    };

    // 1. Create each input→kernel→output group
    for (let i = 0; i < inChannels; i++) {
      nodes = nodes.concat(createChannelSlice(i, nodeId, kernelIdx, fetcherType, sliceConfig));
    }

    const sumConfig: LayoutConfig = {
      childWidth: 120,
      childHeight: 100,
      padding: 10,
      direction: pageDirection,
    };

    // 2. Create sum group
    nodes = nodes.concat(createSumGroup(nodeId, kernelIdx, fetcherType, sumConfig));


    // 3. Create edges between LayerNodes
    const edges = createSliceToSumEdges(inChannels, 'sum-layer');

    return { nodes, edges };
  }, [fetcherType, pageDirection]);

  const { kernelNodes, kernelEdges } = useMemo(() => {
    if (modelData && nodeId && kernelIndex) {
      const result = generateKernelDetailView(modelData, nodeId, parseInt(kernelIndex));
      if (result) {
        return { kernelNodes: result.nodes, kernelEdges: result.edges };
      }
    }
    return { kernelNodes: [], kernelEdges: [] };
  }, [modelData, nodeId, kernelIndex, generateKernelDetailView]);

  const handleBackClick = () => {
    navigate('/');
  };

  if (!modelData) {
    return <div className="flex items-center justify-center h-screen">Loading kernel details...</div>;
  }

  return (
    <SharedCanvas
      nodes={kernelNodes}
      edges={kernelEdges}
      fitView
      minZoom={0.5}
      maxZoom={2}
      pageDirection={pageDirection}
    >
      {/* ── Back button ── */}
      <Panel position="top-left">
        <button
          onClick={handleBackClick}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '7px 12px',
            background: 'rgba(13, 13, 20, 0.88)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 10,
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            color: 'rgba(255,255,255,0.75)',
            fontSize: 11,
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          ← Overview
        </button>
      </Panel>
    </SharedCanvas>
  );
}