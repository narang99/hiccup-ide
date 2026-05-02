import { useEffect } from 'react';
import { useModelDataCacheStore } from '../stores/modelDataCacheStore';
import { type ModelData } from '../types/model';

interface UseModelDataResult {
  modelData: ModelData | null;
  loading: boolean;
  error: string | null;
}

export const useModelData = (modelAlias: string, apiBaseUrl?: string): UseModelDataResult => {
  const { fetchModelData, getModelData } = useModelDataCacheStore();

  const cacheEntry = getModelData(modelAlias);

  useEffect(() => {
    fetchModelData(modelAlias, apiBaseUrl);
  }, [modelAlias, apiBaseUrl, fetchModelData]);

  return {
    modelData: cacheEntry?.status === 'success' ? cacheEntry.data : null,
    loading: cacheEntry?.status === 'loading' || !cacheEntry,
    error: cacheEntry?.status === 'error' ? cacheEntry.error : null,
  };
};