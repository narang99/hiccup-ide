import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ReactFlow, Background, Controls, MiniMap, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { type ModelData } from '../types/model';
import ConvOutActNode from './nodes/ConvOutActNode';
import { DEFAULT_FETCHERS } from '../fetchers';

export default function KernelDetailView() {
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
      nodes.push({
        id: `input-${i}`,
        type: 'default',
        position: { x: 100, y: 200 + i * 100 },
        data: {
          label: (
            <div className="text-center">
              <div className="font-bold text-sm">Input {i}</div>
              <div className="text-xs text-gray-500">Channel {i}</div>
            </div>
          ),
        },
        style: {
          background: '#10b981',
          color: '#fff',
          border: '1px solid #374151',
          borderRadius: '6px',
          padding: '10px',
          minWidth: '80px',
        },
      });

      // Kernel slice nodes
      nodes.push({
        id: `kernel-slice-${i}`,
        type: 'default',
        position: { x: 300, y: 200 + i * 100 },
        data: {
          label: (
            <div className="text-center">
              <div className="font-bold text-sm">K{kernelIdx}:{i}</div>
              <div className="text-xs text-gray-500">
                {targetNode.params.kernel_size ? 
                  `${(targetNode.params.kernel_size as number[])[0]}×${(targetNode.params.kernel_size as number[])[1]}` : 
                  'Kernel'}
              </div>
            </div>
          ),
        },
        style: {
          background: '#8b5cf6',
          color: '#fff',
          border: '1px solid #374151',
          borderRadius: '6px',
          padding: '10px',
          minWidth: '80px',
        },
      });

      // Slice output nodes
      const coordinate = `${nodeId}.out_${kernelIndex}.in_${i}`;
      nodes.push({
        id: `output-${i}`,
        type: 'default',
        position: { x: 500, y: 200 + i * 100 },
        data: {
          label: (
            <ConvOutActNode
              fetchers={DEFAULT_FETCHERS}
              maxSize={84}
              coordinate={coordinate}
              title="Output"
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
    nodes.push({
      id: 'sum',
      type: 'default',
      position: { x: 700, y: 200 + (inChannels / 2) * 100 },
      data: {
        label: (
          <div className="text-center">
            <div className="font-bold text-xl">Σ</div>
            <div className="text-xs text-gray-500">Sum</div>
          </div>
        ),
      },
      style: {
        background: '#ef4444',
        color: '#fff',
        border: '2px solid #374151',
        borderRadius: '8px',
        padding: '15px',
        minWidth: '60px',
      },
    });

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
  }, []);

  useEffect(() => {
    // Load model data
    fetch('/example-model.json')
      .then((response) => response.json())
      .then((data: ModelData) => {
        setModelData(data);
        
        if (nodeId && kernelIndex) {
          generateKernelDetailView(data, nodeId, parseInt(kernelIndex));
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
    <div style={{ width: '100vw', height: '100vh' }}>
      {/* Back button */}
      <button
        onClick={handleBackClick}
        className="absolute top-4 left-4 z-10 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg"
      >
        ← Back to Overview
      </button>
      
      <ReactFlow
        nodes={kernelNodes}
        edges={kernelEdges}
        fitView
        minZoom={0.5}
        maxZoom={2}
      >
        <Controls />
        <MiniMap />
        <Background />
      </ReactFlow>
    </div>
  );
}