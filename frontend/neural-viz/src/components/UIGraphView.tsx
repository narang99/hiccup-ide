import { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getUIGraph } from '../fetchers/graph';
import { getLayoutedUIGraphNodes } from '../utils/graphLayout';
import { getUIGraphNodeText, getUIGraphNodeId } from '../utils/uiGraphStrings';
import type { LinkNode, LinkEdge } from '../fetchers/graph';

const UIGraphView = () => {
  const { modelAlias, inputAlias, workAlias } = useParams();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    const fetchGraph = async () => {
      if (!modelAlias || !inputAlias || !workAlias) return;

      try {
        setLoading(true);
        const data = await getUIGraph(modelAlias, inputAlias, workAlias, startCoordinates);
        
        // Transform backend node-link data to React Flow format
        // LinkNode has { id: UIGraphNode }
        const rfNodes: Node[] = data.nodes.map((node: LinkNode) => {
          const uiNode = node.id;
          return {
            id: getUIGraphNodeId(uiNode),
            type: 'default',
            data: { label: getUIGraphNodeText(uiNode) },
            position: { x: 0, y: 0 }, // Will be set by layout
          };
        });

        console.log(data);
        const rfEdges: Edge[] = data.edges.map((edge: LinkEdge, index: number) => ({
          id: `e-${index}`,
          source: getUIGraphNodeId(edge.source),
          target: getUIGraphNodeId(edge.target),
        }));

        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedUIGraphNodes(rfNodes, rfEdges);
        
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
        setLoading(false);
      } catch (err: any) {
        console.error('Error fetching UI graph:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchGraph();
  }, [modelAlias, inputAlias, workAlias, startCoordinates, setNodes, setEdges]);

  if (loading) {
    return <div className="flex items-center justify-center h-full">Loading graph...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">Error: {error}</div>;
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default UIGraphView;
