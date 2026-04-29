import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, addEdge, type OnConnect, type Node, type Edge } from '@xyflow/react';
import { type ModelData, type ExpandedState } from '../types/model';
import { createChannelNode } from '../utils/nodeUtils';
import { createConvolutionNode } from '../utils/createConvolutionNode';
import { createChannelEdges, createReactFlowEdge } from '../utils/edgeUtils';

export const useModelVisualization = () => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const initialNodesForState: Node[] = []
  const initialEdgesForState: Edge[] = []
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodesForState);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdgesForState);
  const [expandedState, setExpandedState] = useState<ExpandedState>({});

  const toggleExpand = useCallback((nodeId: string) => {
    setExpandedState(prev => ({ ...prev, [nodeId]: !prev[nodeId] }));
  }, []);

  const generateNodesAndEdges = useCallback((data: ModelData, expandedState: ExpandedState, toggleExpandFn: (nodeId: string) => void) => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    // Create main nodes
    data.nodes.forEach((modelNode, index) => {
      const mainNode = createConvolutionNode(modelNode, index, expandedState, toggleExpandFn);
      allNodes.push(mainNode);

      // Add channel nodes if expanded
      if (expandedState[modelNode.id] && modelNode.type === 'Conv2d') {
        const inChannels = modelNode.params.in_channels as number;
        const outChannels = modelNode.params.out_channels as number;

        // Input channels
        for (let i = 0; i < inChannels; i++) {
          allNodes.push(createChannelNode(modelNode, i, true, mainNode.position));
        }

        // Output channels
        for (let i = 0; i < outChannels; i++) {
          allNodes.push(createChannelNode(modelNode, i, false, mainNode.position));
        }
      }
    });

    // Create edges
    data.edges.forEach((modelEdge) => {
      const sourceNode = data.nodes.find(n => n.id === modelEdge.source);
      const targetNode = data.nodes.find(n => n.id === modelEdge.target);

      if (sourceNode && targetNode && 
          sourceNode.type === 'Conv2d' && targetNode.type === 'Conv2d') {
        // Conv2d to Conv2d: use channel-aware edges
        allEdges.push(...createChannelEdges(sourceNode, targetNode, expandedState));
      } else {
        // Regular edge for non-Conv2d connections
        allEdges.push(createReactFlowEdge(modelEdge));
      }
    });

    return { nodes: allNodes, edges: allEdges };
  }, []);

  const onConnect: OnConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Load model data
  useEffect(() => {
    fetch('/example-model.json')
      .then((response) => response.json())
      .then((data: ModelData) => {
        setModelData(data);
      })
      .catch((error) => console.error('Error loading model data:', error));
  }, []);

  // Update nodes and edges when model data or expanded state changes
  useEffect(() => {
    if (modelData) {
      const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(modelData, expandedState, toggleExpand);
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [modelData, expandedState, generateNodesAndEdges, toggleExpand, setNodes, setEdges]);

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
  };
};