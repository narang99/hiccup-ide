import { create } from 'zustand';
import { loadLayerSaliencyMaps, type LayerSaliencyData } from '../fetchers/saliency_map';

interface LayerSaliencyState {
  cachedLayer: string | null;
  saliencyData: LayerSaliencyData | null;
  isLoading: boolean;
  error: string | null;
  
  fetchLayerSaliency: (modelAlias: string, inputAlias: string, layerName: string) => Promise<void>;
  clearCache: () => void;
  getCachedData: (layerName: string) => LayerSaliencyData | null;
}

export const useLayerSaliencyStore = create<LayerSaliencyState>((set, get) => ({
  cachedLayer: null,
  saliencyData: null,
  isLoading: false,
  error: null,
  
  fetchLayerSaliency: async (modelAlias: string, inputAlias: string, layerName: string) => {
    const { cachedLayer, saliencyData } = get();
    
    // Return cached data if we already have it for this layer
    if (cachedLayer === layerName && saliencyData) {
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const data = await loadLayerSaliencyMaps(modelAlias, inputAlias, layerName);
      
      set({
        cachedLayer: layerName,
        saliencyData: data,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch saliency data',
      });
    }
  },
  
  clearCache: () => {
    set({
      cachedLayer: null,
      saliencyData: null,
      isLoading: false,
      error: null,
    });
  },
  
  getCachedData: (layerName: string) => {
    const { cachedLayer, saliencyData } = get();
    return cachedLayer === layerName ? saliencyData : null;
  },
}));