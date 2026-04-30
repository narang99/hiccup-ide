import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, addEdge, type OnConnect, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { createConvolutionNode } from '../utils/createConvolutionNode';
import { 
  createOutputKernelNode
} from '../utils/kernelNodes';
import { 
  createKernelEdges,
  createInputToKernelSliceEdges
} from '../utils/kernelEdges';
import { ReluChannelNodeData } from '../components/nodes/ReluChannelNode';
import { type NodeFetchers, loadActivationFromFile } from '../fetchers';

export const useModelVisualization = () => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  
  // Create fetchers for data loading
  const fetchers: NodeFetchers = {
    activation: loadActivationFromFile
  };
  
  const generateNodesAndEdges = (
    data: ModelData
  ) => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    // Create nodes - kernels for Conv2d, regular nodes for others
    data.nodes.forEach((modelNode, index) => {
      if (modelNode.type === 'Conv2d') {
        // For Conv2d: only show kernels, no main block
        const basePosition = { x: 300, y: index * 200 }; // Virtual position for kernel placement
        const outChannels = modelNode.params.out_channels as number;
        
        for (let kernelIndex = 0; kernelIndex < outChannels; kernelIndex++) {
          const kernelNode = createOutputKernelNode(
            modelNode, 
            kernelIndex, 
            basePosition,
            fetchers
          );
          allNodes.push(kernelNode);
        }
      } else if (modelNode.type === 'ReLU') {
        // For ReLU: show channel-wise operations
        const basePosition = { x: 300, y: index * 200 };
        // Get number of channels from the shape (assuming format [batch, channels, height, width])
        const numChannels = modelNode.shape.length > 1 ? modelNode.shape[1] : 1;
        
        for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
          const reluChannelNode = {
            id: `${modelNode.id}-channel-${channelIndex}`,
            type: 'default',
            position: {
              x: basePosition.x + (channelIndex - numChannels / 2) * 120,
              y: basePosition.y
            },
            data: {
              label: ReluChannelNodeData({ channelIndex, layerId: modelNode.id, fetchers }),
            },
            style: {
              background: '#fbbf24', // Yellow for ReLU
              color: '#000',
              border: '1px solid #374151',
              borderRadius: '6px',
              padding: '8px',
              minWidth: '50px',
              fontSize: '10px',
            },
            parentNode: modelNode.id,
          };
          allNodes.push(reluChannelNode);
        }
      } else {
        // For other layers: show regular node
        const mainNode = createConvolutionNode(modelNode, index);
        allNodes.push(mainNode);
      }
    });

    // Create main edges between layers
    data.edges.forEach((modelEdge) => {
      const sourceNode = data.nodes.find(n => n.id === modelEdge.source);
      const targetNode = data.nodes.find(n => n.id === modelEdge.target);

      if (sourceNode && targetNode) {
        allEdges.push(...createKernelEdges(sourceNode, targetNode));
        allEdges.push(...createInputToKernelSliceEdges());
      }
    });

    return { nodes: allNodes, edges: allEdges };
  };

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

  // Update nodes and edges when model data or kernel expanded state changes
  useEffect(() => {
    if (modelData) {
      const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(modelData);
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [modelData, setNodes, setEdges]);

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
  };
};