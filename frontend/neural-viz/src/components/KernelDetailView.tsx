import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useFetcherType } from '../hooks/useFetcherType';
import SharedCanvas from './SharedCanvas';
import { makeEvenlySpacedHorizontalLayout } from '../layouts/horizontal';

const getNodeShowingActivation = (
  id: string, 
  position: { x: number, y: number }, 
  title: string, 
  coordinate: string, 
  fetcherType: FetcherType = "activation", 
  parentId?: string, 
  width?: number, 
  height?: number,
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

export default function KernelDetailView() {
  const { fetcherType } = useFetcherType();
  const { nodeId, kernelIndex } = useParams<{ nodeId: string; kernelIndex: string }>();
  const navigate = useNavigate();
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [kernelNodes, setKernelNodes] = useState<Node[]>([]);
  const [kernelEdges, setKernelEdges] = useState<Edge[]>([]);

  const generateKernelDetailView = useCallback((data: ModelData, nodeId: string, kernelIdx: number) => {
    const targetNode = data.nodes.find(n => n.id === nodeId);
    if (!targetNode || targetNode.type !== 'Conv2d') return;

    const inChannels = targetNode.params.in_channels as number;
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    // Create each input→kernel→output group as a LayerNode parent
    for (let i = 0; i < inChannels; i++) {
      const sliceLayerId = `slice-${i}`;
      const childWidth = 120;
      const childHeight = 100;
      const padding = 10;
      const sliceLayout = makeEvenlySpacedHorizontalLayout(3, childHeight, childWidth, padding);

      // Create parent LayerNode for this slice
      nodes.push({
        id: sliceLayerId,
        type: 'LayerNode',
        position: { x: 0, y: 0 }, // dagre will position this
        width: sliceLayout.parent.width,
        height: sliceLayout.parent.height,
        data: {
          label: `Channel ${i} Slice`,
          layerType: 'Conv2d',
          nodeCount: 3,
        },
      });

      // Input node
      const inputCoordinate = `${nodeId}.out_${kernelIdx}.in_${i}.input`;
      nodes.push(getNodeShowingActivation(
        `input-${i}`,
        sliceLayout.children[0],
        "Input",
        inputCoordinate,
        fetcherType,
        sliceLayerId,
        childWidth,
        childHeight
      ));

      // Kernel slice node
      const weightCoordinate = `${nodeId}.out_${kernelIdx}.in_${i}`;
      nodes.push(getNodeShowingActivation(
        `kernel-slice-${i}`,
        sliceLayout.children[1],
        `K${kernelIdx}:${i}`,
        weightCoordinate,
        'weight',
        sliceLayerId,
        childWidth,
        childHeight
      ));

      // Slice output node
      const outputCoordinate = `${nodeId}.out_${kernelIdx}.in_${i}`;
      nodes.push(getNodeShowingActivation(
        `output-${i}`,
        sliceLayout.children[2],
        "Output",
        outputCoordinate,
        fetcherType,
        sliceLayerId,
        childWidth,
        childHeight
      ));
    }

    // Create sum LayerNode
    const sumLayerId = 'sum-layer';
    const sumChildWidth = 120;
    const sumChildHeight = 100;
    const sumPadding = 10;
    const sumLayout = makeEvenlySpacedHorizontalLayout(1, sumChildHeight, sumChildWidth, sumPadding);

    nodes.push({
      id: sumLayerId,
      type: 'LayerNode',
      position: { x: 0, y: 0 }, // dagre will position this
      width: sumLayout.parent.width,
      height: sumLayout.parent.height,
      data: {
        label: 'Sum',
        layerType: 'Output',
        nodeCount: 1,
      },
    });

    // Sum activation node
    const sumCoordinate = `${nodeId}.out_${kernelIdx}`;
    nodes.push(getNodeShowingActivation(
      'sum',
      sumLayout.children[0],
      "Sum",
      sumCoordinate,
      fetcherType,
      sumLayerId,
      sumChildWidth,
      sumChildHeight
    ));

    // Create edges between LayerNodes
    for (let i = 0; i < inChannels; i++) {
      edges.push({
        id: `slice-${i}-to-sum`,
        source: `slice-${i}`,
        target: sumLayerId,
        type: 'default',
        style: { stroke: '#dc2626', strokeWidth: 2 },
      });
    }

    setKernelNodes(nodes);
    setKernelEdges(edges);
  }, [fetcherType, kernelIndex]);

  useEffect(() => {
    // Load model data
    const modelAlias = "example-model";
    const apiBaseUrl = "http://localhost:8000";

    fetch(`${apiBaseUrl}/api/models/${modelAlias}/`)
      .then((response) => response.json())
      .then((data: { definition: ModelData }) => {
        const modelData = data.definition;
        setModelData(modelData);

        if (nodeId && kernelIndex) {
          generateKernelDetailView(modelData, nodeId, parseInt(kernelIndex));
        }
      })
      .catch((error) => console.error('Error loading model data:', error));
  }, [nodeId, kernelIndex, generateKernelDetailView]);

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