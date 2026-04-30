import { ReactFlow, Background, Controls, MiniMap, Panel } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useModelVisualization } from '../hooks/useModelVisualization';
import { useColormap } from '../contexts/ColormapContext';
import { COLORMAPS, COLORMAP_META, type ColormapName } from '../utils/colormaps';

const COLORMAP_KEYS = Object.keys(COLORMAPS) as ColormapName[];

export default function ModelVisualization() {
  const {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
  } = useModelVisualization();

  const { colormap: activeColormap, setColormap } = useColormap();

  if (!modelData) {
    return <div className="flex items-center justify-center h-screen">Loading model...</div>;
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background />

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
                  {/* Gradient swatch */}
                  <span style={{
                    display: 'inline-block',
                    width: 28,
                    height: 8,
                    borderRadius: 3,
                    background: meta.gradient,
                    flexShrink: 0,
                  }} />
                  {/* Label */}
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