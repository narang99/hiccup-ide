import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, type Node, type Edge } from '@xyflow/react';
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
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';

export const useModelVisualization = (fetcherType: FetcherType = "activation") => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const generateNodesAndEdges = useCallback((
    data: ModelData
  ) => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    // Create nodes - kernels for Conv2d, regular nodes for others
    data.nodes.forEach((modelNode, index) => {
      if (modelNode.type === 'Conv2d') {
        // For Conv2d: create parent layer node and kernel sub-nodes
        const basePosition = { x: 300, y: index * 200 };
        const outChannels = modelNode.params.out_channels as number;
        
        // Calculate parent node dimensions
        const parentWidth = Math.max(200, (outChannels * 120) + 40);
        const parentHeight = 140;

        // Create parent layer node
        const parentLayerNode = {
          id: modelNode.id,
          type: 'group',
          position: basePosition,
          data: {
            label: `Conv2d Layer (${outChannels} kernels)`,
          },
          style: {
            background: 'rgba(96, 165, 250, 0.1)',
            border: '2px solid rgba(96, 165, 250, 0.3)',
            borderRadius: '8px',
            padding: '20px',
            width: parentWidth,
            height: parentHeight,
          },
        };
        allNodes.push(parentLayerNode);

        // Create kernel sub-nodes with parentId
        for (let kernelIndex = 0; kernelIndex < outChannels; kernelIndex++) {
          const kernelNode = createOutputKernelNode(
            modelNode,
            kernelIndex,
            { x: 20 + (kernelIndex * 120), y: 20 }, // Relative to parent
            DEFAULT_FETCHERS,
            fetcherType
          );
          // Add parentId to the kernel node
          kernelNode.parentId = modelNode.id;
          kernelNode.extent = 'parent';
          allNodes.push(kernelNode);
        }
      } else if (modelNode.type === 'ReLU') {
        // For ReLU: create parent layer node and channel sub-nodes
        const basePosition = { x: 300, y: index * 200 };
        // Get number of channels from the shape (assuming format [batch, channels, height, width])
        const numChannels = modelNode.shape.length > 1 ? modelNode.shape[1] : 1;
        
        // Calculate parent node dimensions
        const parentWidth = Math.max(160, (numChannels * 110) + 40);
        const parentHeight = 120;

        // Create parent layer node
        const parentLayerNode = {
          id: modelNode.id,
          type: 'group',
          position: basePosition,
          data: {
            label: `ReLU Layer (${numChannels} channels)`,
          },
          style: {
            background: 'rgba(251, 191, 36, 0.1)',
            border: '2px solid rgba(251, 191, 36, 0.3)',
            borderRadius: '8px',
            padding: '20px',
            width: parentWidth,
            height: parentHeight,
          },
        };
        allNodes.push(parentLayerNode);

        // Create channel sub-nodes with parentId
        for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
          const reluChannelNode: Node = {
            id: `${modelNode.id}-channel-${channelIndex}`,
            type: 'default',
            position: {
              x: 20 + (channelIndex * 110), // Relative to parent
              y: 20 // Relative to parent
            },
            data: {
              label: ReluChannelNodeData({ channelIndex, layerId: modelNode.id, fetchers: DEFAULT_FETCHERS, fetcherType }),
            },
            style: {
              background: 'transparent',
              border: '1px solid rgba(251, 191, 36, 0.35)',
              borderRadius: '6px',
              padding: 0,
              minWidth: '96px',
              fontSize: '10px',
              overflow: 'hidden',
            },
            parentId: modelNode.id,
            extent: 'parent',
            // draggable: false,
          };
          allNodes.push(reluChannelNode);
        }
      } else {
        // For other layers: show regular node
        const mainNode = createConvolutionNode(modelNode, index);
        allNodes.push(mainNode);
      }
    });

    // Create simple edges between layer groups only
    data.edges.forEach((modelEdge) => {
      allEdges.push({
        id: `${modelEdge.source}-${modelEdge.target}`,
        source: modelEdge.source,
        target: modelEdge.target,
        type: 'default',
      });
    });

    return { nodes: allNodes, edges: allEdges };
  }, [fetcherType]);


  // Load model data
  useEffect(() => {
    const modelAlias = "example-model";
    const apiBaseUrl = "http://localhost:8000";

    fetch(`${apiBaseUrl}/api/models/${modelAlias}/`)
      .then((response) => response.json())
      .then((data) => {
        setModelData(data.definition);
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
  }, [modelData, generateNodesAndEdges, setNodes, setEdges]);

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
  };
};