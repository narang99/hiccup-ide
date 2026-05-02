import { createContext } from 'react';
import { type FetcherType } from '../fetchers';

interface FetcherTypeContextValue {
  fetcherType: FetcherType;
  setFetcherType: (type: FetcherType) => void;
}

export const FetcherTypeContext = createContext<FetcherTypeContextValue>({
  fetcherType: 'saliency_map',
  setFetcherType: () => {},
});