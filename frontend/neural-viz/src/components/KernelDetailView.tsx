import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import ConvOutActNode from './nodes/ConvOutActNode';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';
import { useFetcherType } from '../contexts/FetcherTypeContext';
import SharedCanvas from './SharedCanvas';

const getNodeShowingActivation = (id: string, position: { x: number, y: number }, title: string, coordinate: string, fetcherType: FetcherType = "activation") => {
  return ({
    id: id,
    type: 'default',
    position: position,
    data: {
      label: (
        <ConvOutActNode
          fetchers={DEFAULT_FETCHERS}
          fetcherType={fetcherType}
          maxSize={84}
          coordinate={coordinate}
          title={title}
        />
      ),
    },
    style: {
      background: 'transparent',
      border: '1px solid rgba(245, 158, 11, 0.35)',
      borderRadius: '6px',
      padding: 0,
      minWidth: '96px',
      overflow: 'hidden',
    },
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

    // Title node
    nodes.push({
      id: 'title',
      type: 'default',
      position: { x: 400, y: 50 },
      data: {
        label: (
          <div className="text-center">
            <div className="font-bold text-lg">{targetNode.id} - Kernel {kernelIdx}</div>
            <div className="text-sm text-gray-600">
              {inChannels} input channels → 1 output channel
            </div>
          </div>
        ),
      },
      style: {
        background: '#f3f4f6',
        border: '2px solid #374151',
        borderRadius: '8px',
        padding: '15px',
        minWidth: '200px',
      },
    });

    // Input channel nodes
    for (let i = 0; i < inChannels; i++) {
      const inputCoordinate = `${nodeId}.out_${kernelIndex}.in_${i}.input`;
      nodes.push(getNodeShowingActivation(`input-${i}`, { x: 100, y: 200 + i * 100 }, "Input", inputCoordinate, fetcherType));

      // Kernel slice nodes
      const weightCoordinate = `${nodeId}.out_${kernelIndex}.in_${i}`;
      nodes.push(getNodeShowingActivation(`kernel-slice-${i}`, { x: 300, y: 200 + i * 100 }, `K${kernelIdx}:${i}`, weightCoordinate, 'weight'));


      // Slice output nodes
      const outputCoordinate = `${nodeId}.out_${kernelIndex}.in_${i}`;
      nodes.push(getNodeShowingActivation(`output-${i}`, { x: 500, y: 200 + i * 100 }, "Output", outputCoordinate, fetcherType));

      // Edges: Input -> Kernel -> Output
      edges.push({
        id: `input-${i}-kernel-${i}`,
        source: `input-${i}`,
        target: `kernel-slice-${i}`,
        type: 'default',
        style: { stroke: '#059669', strokeWidth: 2 },
      });

      edges.push({
        id: `kernel-${i}-output-${i}`,
        source: `kernel-slice-${i}`,
        target: `output-${i}`,
        type: 'default',
        style: { stroke: '#7c3aed', strokeWidth: 2 },
      });
    }

    // Sum node
    const sumCoordinate = `${nodeId}.out_${kernelIndex}`;
    nodes.push(getNodeShowingActivation(`sum`, { x: 700, y: 200 + (inChannels / 2) * 100 }, "sum", sumCoordinate, fetcherType));

    // Edges from all outputs to sum
    for (let i = 0; i < inChannels; i++) {
      edges.push({
        id: `output-${i}-sum`,
        source: `output-${i}`,
        target: 'sum',
        type: 'default',
        style: { stroke: '#dc2626', strokeWidth: 2 },
      });
    }

    setKernelNodes(nodes);
    setKernelEdges(edges);
  }, [fetcherType]);

  useEffect(() => {
    // Load model data
    const modelAlias = "example-model";
    const apiBaseUrl = "http://localhost:8000";
    
    fetch(`${apiBaseUrl}/api/models/${modelAlias}/`)
      .then((response) => response.json())
      .then((data: any) => {
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