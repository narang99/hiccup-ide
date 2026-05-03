import { useMemo } from 'react';
import { type Node, type Edge } from '@xyflow/react';
import { useModelData } from './useModelData';
import { createConv2dLayer } from '../layerCreators/conv';
import { createReLULayer } from '../layerCreators/relu';
import { createInputLayer } from '../layerCreators/input';
import { createOtherLayer } from '../layerCreators/default';
import { type FetcherType } from '../fetchers';
import { type Direction } from '../types/direction';

export const useSingleLayer = (
  modelAlias: string,
  layerId: string,
  fetcherType: FetcherType = "activation",
  pageDirection: Direction = "TB"
) => {
  const { modelData, loading, error } = useModelData(modelAlias);

  const { nodes, edges } = useMemo(() => {
    if (!modelData) return { nodes: [], edges: [] };

    const modelNode = modelData.nodes.find(n => n.id === layerId);
    if (!modelNode) {
      return { nodes: [], edges: [] };
    }

    // Single layer at origin
    const basePosition = { x: 0, y: 0 };
    
    // Hardcoded directions for single layer view as TB/LR
    const layerBlockHandleDirection = pageDirection;
    const directionInsideLayerBlock = pageDirection === "TB" ? "LR" : "TB";

    let layerNodes: Node[];
    
    switch (modelNode.type) {
      case 'Conv2d':
        layerNodes = createConv2dLayer(
          modelNode, 
          basePosition, 
          fetcherType, 
          layerBlockHandleDirection, 
          directionInsideLayerBlock
        );
        break;
      case 'ReLU':
        layerNodes = createReLULayer(
          modelNode, 
          basePosition, 
          fetcherType, 
          layerBlockHandleDirection, 
          directionInsideLayerBlock
        );
        break;
      case 'Input':
        layerNodes = createInputLayer(
          modelNode, 
          basePosition, 
          fetcherType, 
          layerBlockHandleDirection, 
          directionInsideLayerBlock
        );
        break;
      default:
        layerNodes = createOtherLayer(
          modelNode, 
          basePosition, 
          layerBlockHandleDirection
        );
        break;
    }

    return {
      nodes: layerNodes,
      edges: [] as Edge[]
    };
  }, [modelData, layerId, fetcherType, pageDirection]);

  return {
    nodes,
    edges,
    loading,
    error,
    modelNode: modelData?.nodes.find(n => n.id === layerId)
  };
};
