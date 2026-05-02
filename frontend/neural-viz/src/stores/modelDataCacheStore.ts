import { create } from 'zustand';
import { type ModelData } from '../types/model';

interface ModelDataCacheEntry {
  data: ModelData;
  status: 'success';
}

interface ModelDataCacheEntryLoading {
  data: null;
  status: 'loading';
}

interface ModelDataCacheEntryError {
  data: null;
  status: 'error';
  error: string;
}

type CacheEntry = ModelDataCacheEntry | ModelDataCacheEntryLoading | ModelDataCacheEntryError;

interface ModelDataCacheState {
  cache: Record<string, CacheEntry>;
  fetchModelData: (modelAlias: string, apiBaseUrl?: string) => Promise<void>;
  getModelData: (modelAlias: string) => CacheEntry | undefined;
  clearCache: () => void;
}

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const useModelDataCacheStore = create<ModelDataCacheState>((set, get) => ({
  cache: {},

  getModelData: (modelAlias: string) => {
    return get().cache[modelAlias];
  },

  fetchModelData: async (modelAlias: string, apiBaseUrl = DEFAULT_API_BASE_URL) => {
    const { cache } = get();
    
    // If already cached and successful, return immediately
    const existing = cache[modelAlias];
    if (existing && existing.status === 'success') {
      return;
    }
    
    // If already loading, return (prevents duplicate requests)
    if (existing && existing.status === 'loading') {
      return;
    }

    // Set loading state
    set((state) => ({
      cache: {
        ...state.cache,
        [modelAlias]: {
          data: null,
          status: 'loading',
        },
      },
    }));

    try {
      const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Store successful result
      set((state) => ({
        cache: {
          ...state.cache,
          [modelAlias]: {
            data: data.definition,
            status: 'success',
          },
        },
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Store error result
      set((state) => ({
        cache: {
          ...state.cache,
          [modelAlias]: {
            data: null,
            status: 'error',
            error: errorMessage,
          },
        },
      }));
    }
  },

  clearCache: () => {
    set({ cache: {} });
  },
}));