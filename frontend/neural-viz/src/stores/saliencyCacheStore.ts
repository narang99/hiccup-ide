import { create } from 'zustand';
import { loadLayerSaliencyMaps, loadBatchSaliencyMaps, type LayerSaliencyData } from '../fetchers/saliency_map';

interface SaliencyCacheState {
  cachedLayer: string | null;
  cachedWorkAlias: string | null;
  cachedPruned: boolean | null;
  saliencyData: LayerSaliencyData | null;
  isLoading: boolean;
  error: string | null;

  fetchAndCacheLayerSaliency: (modelAlias: string, inputAlias: string, layerName: string, workAlias?: string, pruned?: boolean) => Promise<LayerSaliencyData>;
  fetchAndCacheBatchSaliency: (modelAlias: string, inputAlias: string, coordinates: string[], cacheKey: string, workAlias?: string, pruned?: boolean) => Promise<LayerSaliencyData>;
  getCachedData: (layerName: string, workAlias?: string, pruned?: boolean) => LayerSaliencyData | null;
  clearCache: () => void;
}

export const useSaliencyCacheStore = create<SaliencyCacheState>((set, get) => ({
  cachedLayer: null,
  cachedWorkAlias: null,
  cachedPruned: null,
  saliencyData: null,
  isLoading: false,
  error: null,

  fetchAndCacheLayerSaliency: async (modelAlias: string, inputAlias: string, layerName: string, workAlias?: string, pruned?: boolean) => {
    const { cachedLayer, cachedWorkAlias, cachedPruned, saliencyData } = get();

    // Return cached data if we already have it for this layer and work context
    if (cachedLayer === layerName && cachedWorkAlias === (workAlias || null) && cachedPruned === (pruned || false) && saliencyData) {
      return saliencyData;
    }

    set({ isLoading: true, error: null });

    try {
      console.log(`Fetching saliency data for layer: ${layerName} (work: ${workAlias}, pruned: ${pruned})`);
      const data = await loadLayerSaliencyMaps(modelAlias, inputAlias, layerName, workAlias, pruned);

      set({
        cachedLayer: layerName,
        cachedWorkAlias: workAlias || null,
        cachedPruned: pruned || false,
        saliencyData: data,
        isLoading: false,
        error: null,
      });

      return data;
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch saliency data',
      });
      throw error;
    }
  },

  fetchAndCacheBatchSaliency: async (modelAlias: string, inputAlias: string, coordinates: string[], cacheKey: string, workAlias?: string, pruned?: boolean) => {
    const { cachedLayer, cachedWorkAlias, cachedPruned, saliencyData } = get();

    // Return cached data if we already have it for this cacheKey and work context
    if (cachedLayer === cacheKey && cachedWorkAlias === (workAlias || null) && cachedPruned === (pruned || false) && saliencyData) {
      return saliencyData;
    }

    set({ isLoading: true, error: null });

    try {
      console.log(`Fetching batch saliency data for: ${cacheKey} (work: ${workAlias}, pruned: ${pruned})`);
      const data = await loadBatchSaliencyMaps(modelAlias, inputAlias, coordinates, workAlias, pruned);

      set({
        cachedLayer: cacheKey,
        cachedWorkAlias: workAlias || null,
        cachedPruned: pruned || false,
        saliencyData: data,
        isLoading: false,
        error: null,
      });

      return data;
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch batch saliency data',
      });
      throw error;
    }
  },

  getCachedData: (layerName: string, workAlias?: string, pruned?: boolean) => {
    const { cachedLayer, cachedWorkAlias, cachedPruned, saliencyData } = get();
    if (cachedLayer === layerName && cachedWorkAlias === (workAlias || null) && cachedPruned === (pruned || false)) {
      return saliencyData;
    }
    return null;
  },

  clearCache: () => {
    set({
      cachedLayer: null,
      cachedWorkAlias: null,
      cachedPruned: null,
      saliencyData: null,
      isLoading: false,
      error: null,
    });
  },
}));