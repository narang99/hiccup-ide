import { useState, useCallback, useEffect } from 'react';
import { useNodesState, useEdgesState, type Node, type Edge } from '@xyflow/react';
import { type FetcherType } from '../fetchers';
import { createConv2dLayer } from '../layerCreators/conv';
import { createReLULayer } from '../layerCreators/relu';
import { createOtherLayer } from '../layerCreators/default';
import { toggleDirection, type Direction } from '../types/direction';
import { useColormap } from './useColormap';
import { useModelData } from './useModelData';
import { useLayerStats } from './useLayerStats';
import { updateNodeAbsMax } from '../utils/nodeUpdates';
import { type ModelData } from '../types/model';

export const useModelVisualization = (fetcherType: FetcherType = "activation", directionOfPage: Direction = "LR") => {
  const { modelData } = useModelData("example-model");
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



  // Update nodes and edges when model data changes
  useEffect(() => {
    if (modelData) {
      const { nodes: newNodes, edges: newEdges } = generateNodesAndEdges(
        modelData, 
        {} // absMax will be applied separately to preserve selection state
      );
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [modelData, generateNodesAndEdges, setNodes, setEdges, scalingMode]);

  // Separate effect for updating absMax to preserve selection state
  useEffect(() => {
    if (scalingMode === 'global' && Object.keys(layerAbsMax).length > 0) {
      setNodes((currentNodes) => updateNodeAbsMax(currentNodes, layerAbsMax));
    }
  }, [layerAbsMax, scalingMode, setNodes]);

  // Handle global scaling using custom hook
  useLayerStats({
    nodes,
    fetcherType,
    scalingMode,
    setLayerAbsMax,
  });

  return {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
  };
};