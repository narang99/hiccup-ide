import { useState, useEffect, type ReactNode } from 'react';
import { type FetcherType } from '../fetchers';
import { FetcherTypeContext } from './FetcherTypeContextDefinition';

const FETCH_TYPE_STORAGE_KEY = 'hiccup-ide-fetcher-type';

const getStoredFetcherType = (): FetcherType => {
  try {
    const stored = localStorage.getItem(FETCH_TYPE_STORAGE_KEY);
    if (stored === 'activation' || stored === 'saliency_map' || stored === 'weight') {
      return stored as FetcherType;
    }
  } catch {
    // localStorage not available or error reading
  }
  return 'saliency_map';
};

export const FetcherTypeProvider = ({ children }: { children: ReactNode }) => {
  const [fetcherType, setFetcherType] = useState<FetcherType>(getStoredFetcherType);

  useEffect(() => {
    try {
      localStorage.setItem(FETCH_TYPE_STORAGE_KEY, fetcherType);
    } catch {
      // localStorage not available or quota exceeded
    }
  }, [fetcherType]);

  return (
    <FetcherTypeContext.Provider value={{ fetcherType, setFetcherType }}>
      {children}
    </FetcherTypeContext.Provider>
  );
};

