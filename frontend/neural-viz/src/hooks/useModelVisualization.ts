import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, type Node, type Edge } from '@xyflow/react';
import { type ModelData } from '../types/model';
import { type FetcherType } from '../fetchers';
import { createConv2dLayer } from '../layerCreators/conv';
import { createReLULayer } from '../layerCreators/relu';
import { createOtherLayer } from '../layerCreators/default';
import { toggleDirection, type Direction } from '../types/direction';
import { useColormap } from './useColormap';
import { fetchActivationsStats, fetchSaliencyMapsStats } from '../fetchers/stats';

export const useModelVisualization = (fetcherType: FetcherType = "activation", directionOfPage: Direction = "LR") => {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [layerAbsMax, setLayerAbsMax] = useState<Record<string, number>>({});
  const { scalingMode } = useColormap();

  // if the page is vertical, we want the layer groups to be kept vertically
  // inside layer groups then, we want horizontal directions
  const directionInsideLayerBlock = toggleDirection(directionOfPage)
  const layerBlockHandleDirection = directionOfPage;

  const generateNodesAndEdges = useCallback((
    data: ModelData,
    absMaxMap: Record<string, number> = {}
  ) => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    // Create nodes for each layer type
    data.nodes.forEach((modelNode, index) => {
      const basePosition = { x: 300, y: index * 200 };
      const absMax = absMaxMap[modelNode.id];
      
      let layerNodes: Node[];
      
      switch (modelNode.type) {
        case 'Conv2d':
          layerNodes = createConv2dLayer(modelNode, basePosition, fetcherType, layerBlockHandleDirection, directionInsideLayerBlock, absMax);
          break;
        case 'ReLU':
          layerNodes = createReLULayer(modelNode, basePosition, fetcherType, layerBlockHandleDirection, directionInsideLayerBlock, absMax);
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
      const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(
        modelData, 
        scalingMode === 'global' ? layerAbsMax : {}
      );
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [modelData, generateNodesAndEdges, setNodes, setEdges, layerAbsMax, scalingMode]);

  // Handle global scaling
  useEffect(() => {
    if (scalingMode !== 'global' || nodes.length === 0) {
      return;
    }

    const modelAlias = "example-model";
    const inputAlias = "first-input";

    const fetchStatsForLayers = async () => {
      const layerNodes = nodes.filter(n => n.type === 'LayerNode');
      const updates: Record<string, number> = {};
      
      for (const layerNode of layerNodes) {
        const childNodes = nodes.filter(n => n.parentId === layerNode.id && n.type === 'ActivationFlowNode');
        if (childNodes.length === 0) continue;

        const coordinates = childNodes.map(n => n.data.coordinate as string);
        
        try {
          const stats = fetcherType === 'activation' 
            ? await fetchActivationsStats(modelAlias, inputAlias, coordinates)
            : await fetchSaliencyMapsStats(modelAlias, inputAlias, coordinates);
          
          const absMax = Math.max(Math.abs(stats.min), Math.abs(stats.max));
          updates[layerNode.id] = absMax;
        } catch (error) {
          console.error(`Failed to fetch stats for layer ${layerNode.id}:`, error);
        }
      }

      if (Object.keys(updates).length > 0) {
        setLayerAbsMax(updates);
      }
    };

    fetchStatsForLayers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scalingMode, fetcherType, nodes.length]); // We use nodes.length to trigger after initial load

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
  };
};