import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { type FetcherType } from '../fetchers';
import { createConv2dLayer } from '../layerCreators/conv';
import { createReLULayer } from '../layerCreators/relu';
import { createOtherLayer } from '../layerCreators/default';
import { toggleDirection, type Direction } from '../types/direction';

export const useModelVisualization = (fetcherType: FetcherType = "activation", directionOfPage: Direction = "LR") => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // if the page is vertical, we want the layer groups to be kept vertically
  // inside layer groups then, we want horizontal directions
  const directionInsideLayerBlock = toggleDirection(directionOfPage)
  const layerBlockHandleDirection = directionOfPage;

  const generateNodesAndEdges = useCallback((
    data: ModelData
  ) => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    // Create nodes for each layer type
    data.nodes.forEach((modelNode, index) => {
      const basePosition = { x: 300, y: index * 200 };
      
      let layerNodes: Node[];
      
      switch (modelNode.type) {
        case 'Conv2d':
          layerNodes = createConv2dLayer(modelNode, basePosition, fetcherType, layerBlockHandleDirection, directionInsideLayerBlock);
          break;
        case 'ReLU':
          layerNodes = createReLULayer(modelNode, basePosition, fetcherType, layerBlockHandleDirection, directionInsideLayerBlock);
          break;
        default:
          layerNodes = createOtherLayer(modelNode, basePosition, layerBlockHandleDirection);
          break;
      }
      
      allNodes.push(...layerNodes);
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
  }, [fetcherType, directionInsideLayerBlock, layerBlockHandleDirection]);


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