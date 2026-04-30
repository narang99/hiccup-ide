import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, addEdge, type OnConnect, type Node, type Edge } from '@xyflow/react';
import { type ModelData, type KernelExpandedState } from '../types/model';
import { createConvolutionNode } from '../utils/createConvolutionNode';
import { 
  createOutputKernelNode
} from '../utils/kernelNodes';
import { 
  createKernelEdges,
  createInputToKernelSliceEdges
} from '../utils/kernelEdges';
import { ReluChannelNodeData } from '../components/nodes/ReluChannelNode';

export const useModelVisualization = () => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [kernelExpandedState, setKernelExpandedState] = useState<KernelExpandedState>({});

  const toggleKernelExpand = (kernelId: string) => {
    setKernelExpandedState(prev => ({ ...prev, [kernelId]: !prev[kernelId] }));
  };

  const generateNodesAndEdges = (
    data: ModelData, 
    kernelExpandedState: KernelExpandedState, 
    toggleKernelExpandFn: (kernelId: string) => void
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
            kernelExpandedState,
            toggleKernelExpandFn,
            basePosition
          );
          allNodes.push(kernelNode);
          
          // TEMPORARILY DISABLED: Add kernel slice nodes if this kernel is expanded
          // const kernelId = `${modelNode.id}-kernel-${kernelIndex}`;
          // if (kernelExpandedState[kernelId]) {
          //   const inChannels = modelNode.params.in_channels as number;
          //   
          //   for (let inputIndex = 0; inputIndex < inChannels; inputIndex++) {
          //     // Input slice node
          //     allNodes.push(createInputSliceNode(
          //       modelNode, kernelIndex, inputIndex, kernelNode.position
          //     ));
          //     
          //     // Kernel slice node
          //     allNodes.push(createKernelSliceNode(
          //       modelNode, kernelIndex, inputIndex, kernelNode.position
          //     ));
          //     
          //     // Slice output node
          //     allNodes.push(createSliceOutputNode(
          //       modelNode, kernelIndex, inputIndex, kernelNode.position
          //     ));
          //   }
          //   
          //   // Sum node
          //   allNodes.push(createSumNode(modelNode, kernelIndex, kernelNode.position));
          //   
          //   // Add internal kernel slice edges
          //   allEdges.push(...createKernelSliceEdges(modelNode, kernelIndex, kernelExpandedState));
          // }
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
              label: ReluChannelNodeData({ channelIndex }),
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
        allEdges.push(...createInputToKernelSliceEdges(sourceNode, targetNode, kernelExpandedState));
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
      const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(
        modelData, 
        kernelExpandedState, 
        toggleKernelExpand
      );
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [modelData, kernelExpandedState, setNodes, setEdges]);

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
  };
};