import { create } from 'zustand';
import { loadLayerSaliencyMaps, type LayerSaliencyData } from '../fetchers/saliency_map';

interface SaliencyCacheState {
  cachedLayer: string | null;
  saliencyData: LayerSaliencyData | null;
  isLoading: boolean;
  error: string | null;
  
  fetchAndCacheLayerSaliency: (modelAlias: string, inputAlias: string, layerName: string) => Promise<LayerSaliencyData>;
  getCachedData: (layerName: string) => LayerSaliencyData | null;
  clearCache: () => void;
}

export const useSaliencyCacheStore = create<SaliencyCacheState>((set, get) => ({
  cachedLayer: null,
  saliencyData: null,
  isLoading: false,
  error: null,
  
  fetchAndCacheLayerSaliency: async (modelAlias: string, inputAlias: string, layerName: string) => {
    const { cachedLayer, saliencyData } = get();
    
    // Return cached data if we already have it for this layer
    if (cachedLayer === layerName && saliencyData) {
      return saliencyData;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      console.log(`Fetching saliency data for layer: ${layerName}`);
      const data = await loadLayerSaliencyMaps(modelAlias, inputAlias, layerName);
      
      set({
        cachedLayer: layerName,
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
  
  getCachedData: (layerName: string) => {
    const { cachedLayer, saliencyData } = get();
    return cachedLayer === layerName ? saliencyData : null;
  },
  
  clearCache: () => {
    set({
      cachedLayer: null,
      saliencyData: null,
      isLoading: false,
      error: null,
    });
  },
}));