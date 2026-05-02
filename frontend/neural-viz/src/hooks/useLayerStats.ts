import { useEffect } from 'react';
import { type Node } from '@xyflow/react';
import { type FetcherType } from '../fetchers';
import { fetchActivationsStats, fetchSaliencyMapsStats } from '../fetchers/stats';

interface UseLayerStatsParams {
  nodes: Node[];
  fetcherType: FetcherType;
  scalingMode: string;
  setLayerAbsMax: (updates: Record<string, number>) => void;
  modelAlias?: string;
  inputAlias?: string;
}

export const useLayerStats = ({
  nodes,
  fetcherType,
  scalingMode,
  setLayerAbsMax,
  modelAlias = "example-model",
  inputAlias = "first-input"
}: UseLayerStatsParams) => {
  useEffect(() => {
    if (scalingMode !== 'global' || nodes.length === 0) {
      return;
    }

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
  }, [inputAlias, modelAlias, scalingMode, fetcherType, nodes.length, setLayerAbsMax]); // Use nodes.length to prevent infinite loops from node object recreation
};