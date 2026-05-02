import { type ReactNode } from 'react';
import { ReactFlow, Background, Controls, Panel, Position, type ReactFlowProps, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { type FetcherType } from '../fetchers';
import { COLORMAPS, COLORMAP_META, type ColormapName } from '../utils/colormaps';
import { useFetcherType } from '../hooks/useFetcherType';
import { useColormap } from '../hooks/useColormap';
import { useSelectedNodeStore } from '../stores/selectedNodeStore';
import { LayerNode } from './nodes/LayerNode';
import { ActivationFlowNode } from './nodes/ActivationFlowNode';
import { LayerSettings } from './LayerSettings';
import dagre from '@dagrejs/dagre';
import type { Direction } from '../types/direction';

const COLORMAP_KEYS = Object.keys(COLORMAPS) as ColormapName[];

const nodeTypes = {
  LayerNode,
  ActivationFlowNode,
};

interface SharedCanvasProps extends ReactFlowProps {
  children?: ReactNode;
  pageDirection?: Direction;
}

export default function SharedCanvas({ children, pageDirection, ...props }: SharedCanvasProps) {
  const { colormap: activeColormap, setColormap, scalingMode, setScalingMode } = useColormap();
  const { fetcherType, setFetcherType } = useFetcherType();
  const { handleNodeClick, handlePaneClick } = useSelectedNodeStore();
  const { nodes, edges } = getLayoutedElements(props, pageDirection);
  const layoutedProps: ReactFlowProps = { ...props, nodes, edges }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow {...layoutedProps} nodeTypes={nodeTypes} onNodeClick={handleNodeClick} onPaneClick={handlePaneClick}>
        <Controls />
        <Background />

        <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
          {/* ── Data Type selector ── */}
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
              Data
            </span>

            {(['activation', 'saliency_map'] as FetcherType[]).map((type) => {
              const isActive = type === fetcherType;
              return (
                <button
                  key={type}
                  onClick={() => setFetcherType(type)}
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
                    fontSize: 10,
                    fontWeight: isActive ? 700 : 400,
                    color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
                    lineHeight: 1,
                  }}>
                    {type === 'activation' ? 'Activations' : 'Saliency Maps'}
                  </span>
                </button>
              );
            })}
          </div>

          {/* ── Colormap selector ── */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            padding: '7px 10px',
            background: 'rgba(13, 13, 20, 0.88)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 10,
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          }}>
            {/* First row: Colormap selection */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'rgba(255,255,255,0.35)',
                marginRight: 2,
                minWidth: '60px',
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

            {/* Second row: Scaling selection */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'rgba(255,255,255,0.35)',
                marginRight: 2,
                minWidth: '60px',
              }}>
                Scaling
              </span>

              {(['local', 'global'] as const).map((mode) => {
                const isActive = mode === scalingMode;
                return (
                  <button
                    key={mode}
                    onClick={() => setScalingMode(mode)}
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
                      flex: 1,
                      justifyContent: 'center',
                    }}
                  >
                    <span style={{
                      fontSize: 10,
                      fontWeight: isActive ? 700 : 400,
                      color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
                      lineHeight: 1,
                      textTransform: 'capitalize',
                    }}>
                      {mode}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* ── Layer Settings Panel ── */}
          <LayerSettings />
        </Panel>

        {children}
      </ReactFlow>
    </div>
  );
}


const getLayoutedElements = (props: SharedCanvasProps, pageDirection?: Direction): { nodes: Node[], edges: Edge[] } => {
  const nodes = props.nodes;
  const edges = props.edges;
  const direction: Direction = (pageDirection === undefined) ? "LR" : pageDirection;

  if (nodes === undefined || edges === undefined) {
    return { nodes: [], edges: [] }
  }

  const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

  const isHorizontal = false;
  dagreGraph.setGraph({ rankdir: direction });

  // Only add LayerNode types to Dagre for positioning
  nodes.forEach((node) => {
    if (node.type === 'LayerNode') {
      dagreGraph.setNode(node.id, { width: node.width, height: node.height });
    }
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const newNodes = nodes.map((node) => {
    if (node.type === 'LayerNode') {
      const nodeWithPosition = dagreGraph.node(node.id);
      const width = node.width;
      const height = node.height;
      if (width === undefined || height === undefined) {
        throw Error(`invalid node ${node}: width and height cannot be undefined, width=${width} height=${height}`);
      }
      const actualWidth = typeof width === 'string' ? parseInt(width) : width;
      const actualHeight = typeof height === 'string' ? parseInt(height) : height;

      return {
        ...node,
        targetPosition: isHorizontal ? Position.Left : Position.Top,
        sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
        position: {
          x: nodeWithPosition.x - actualWidth / 2,
          y: nodeWithPosition.y - actualHeight / 2,
        },
      };
    } else {
      return node;
    }
  });

  return { nodes: newNodes, edges };
};