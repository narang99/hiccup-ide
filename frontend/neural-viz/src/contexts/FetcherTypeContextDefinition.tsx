import { createContext } from 'react';
import { type FetcherType } from '../fetchers';

interface FetcherTypeContextValue {
  fetcherType: FetcherType;
  setFetcherType: (type: FetcherType) => void;
  workAlias: string | null;
  graphAlias: string | null;
  setWorkGraph: (work: string | null, graph: string | null) => void;
}

export const FetcherTypeContext = createContext<FetcherTypeContextValue>({
  fetcherType: 'saliency_map',
  setFetcherType: () => {},
  workAlias: null,
  graphAlias: null,
  setWorkGraph: () => {},
});