import { type ReactNode } from 'react';
import { ReactFlow, Background, Controls, MiniMap, Panel, Position, type ReactFlowProps, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { type FetcherType } from '../fetchers';
import { COLORMAPS, COLORMAP_META, type ColormapName } from '../utils/colormaps';
import { useFetcherType } from '../hooks/useFetcherType';
import { useColormap } from '../hooks/useColormap';
import { LayerNode } from './nodes/LayerNode';
import dagre from '@dagrejs/dagre';

const COLORMAP_KEYS = Object.keys(COLORMAPS) as ColormapName[];

const nodeTypes = {
  LayerNode,
};

interface SharedCanvasProps extends ReactFlowProps {
  children?: ReactNode;
}

export default function SharedCanvas({ children, ...props }: SharedCanvasProps) {
  const { colormap: activeColormap, setColormap } = useColormap();
  const { fetcherType, setFetcherType } = useFetcherType();
  const { nodes, edges } = getLayoutedElements(props, 172, 36);
  const layoutedProps: ReactFlowProps = { ...props, nodes, edges }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow {...layoutedProps} nodeTypes={nodeTypes}>
        <Controls />
        <MiniMap />
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

        {children}
      </ReactFlow>
    </div>
  );
}


const getLayoutedElements = (props: ReactFlowProps, nodeWidth = 172, nodeHeight = 36): { nodes: Node[], edges: Edge[] } => {
  const nodes = props.nodes;
  const edges = props.edges;
  const direction = "TB";
  if (nodes === undefined || edges === undefined) {
    return { nodes: [], edges: [] }
  }

  const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

  const isHorizontal = false;
  dagreGraph.setGraph({ rankdir: direction, nodesep: 100 });

  // Only add LayerNode types to Dagre for positioning
  nodes.forEach((node) => {
    if (node.type === 'LayerNode') {
      const width = node.style?.width || nodeWidth;
      const height = node.style?.height || nodeHeight;
      dagreGraph.setNode(node.id, { 
        width: typeof width === 'string' ? parseInt(width) : width,
        height: typeof height === 'string' ? parseInt(height) : height
      });
    }
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const newNodes = nodes.map((node) => {
    if (node.type === 'LayerNode') {
      // LayerNodes get positioned by Dagre
      const nodeWithPosition = dagreGraph.node(node.id);
      const width = node.style?.width || nodeWidth;
      const height = node.style?.height || nodeHeight;
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
    } else if (node.parentId) {
      // Child nodes keep their manual relative positions
      return node;
      // return {
      //   ...node,
      //   // targetPosition: isHorizontal ? Position.Left : Position.Top,
      //   // sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      // };
    } else {
      // Other nodes keep their original positions
      return node;
      // return {
      //   ...node,
      //   // targetPosition: isHorizontal ? Position.Left : Position.Top,
      //   // sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      // };
    }
  });

  return { nodes: newNodes, edges };
};