import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ReactFlow, Background, Controls, MiniMap, Panel, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { type ModelData } from '../types/model';
import ConvOutActNode from './nodes/ConvOutActNode';
import { DEFAULT_FETCHERS } from '../fetchers';
import { useColormap } from '../contexts/ColormapContext';
import { COLORMAPS, COLORMAP_META, type ColormapName } from '../utils/colormaps';

const COLORMAP_KEYS = Object.keys(COLORMAPS) as ColormapName[];

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
      const inputCoordinate = `${nodeId}.out_${kernelIndex}.in_${i}.input`;
      nodes.push({
        id: `input-${i}`,
        type: 'default',
        // position: { x: 500, y: 200 + i * 100 },
        position: { x: 100, y: 200 + i * 100 },
        data: {
          label: (
            <ConvOutActNode
              fetchers={DEFAULT_FETCHERS}
              maxSize={84}
              coordinate={inputCoordinate}
              title="Input"
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
      const outputCoordinate = `${nodeId}.out_${kernelIndex}.in_${i}`;
      nodes.push({
        id: `output-${i}`,
        type: 'default',
        position: { x: 500, y: 200 + i * 100 },
        data: {
          label: (
            <ConvOutActNode
              fetchers={DEFAULT_FETCHERS}
              maxSize={84}
              coordinate={outputCoordinate}
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

  const { colormap: activeColormap, setColormap } = useColormap();

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
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

        {/* ── Colormap selector ── */}
        <Panel position="top-right">
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '7px 10px',
            background: 'rgba(13, 13, 20, 0.88)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 10,
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          }}>
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'rgba(255,255,255,0.35)',
              marginRight: 2,
            }}>
              Colormap
            </span>

            {COLORMAP_KEYS.map((key) => {
              const isActive = key === activeColormap;
              const meta = COLORMAP_META[key];
              return (
                <button
                  key={key}
                  onClick={() => setColormap(key)}
                  title={key}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '4px 9px',
                    borderRadius: 6,
                    border: isActive
                      ? '1px solid rgba(255,255,255,0.3)'
                      : '1px solid rgba(255,255,255,0.07)',
                    background: isActive
                      ? 'rgba(255,255,255,0.12)'
                      : 'transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span style={{
                    display: 'inline-block',
                    width: 28,
                    height: 8,
                    borderRadius: 3,
                    background: meta.gradient,
                    flexShrink: 0,
                  }} />
                  <span style={{
                    fontSize: 10,
                    fontWeight: isActive ? 700 : 400,
                    color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
                    lineHeight: 1,
                  }}>
                    {meta.label}
                  </span>
                </button>
              );
            })}
          </div>
        </Panel>

      </ReactFlow>
    </div>
  );
}