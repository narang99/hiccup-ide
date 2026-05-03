import { useState, type ReactNode } from 'react';
import { type FetcherType } from '../fetchers';
import { FetcherTypeContext } from './FetcherTypeContextDefinition';

export const FetcherTypeProvider = ({ children }: { children: ReactNode }) => {
  const [fetcherType, setFetcherType] = useState<FetcherType>('saliency_map');
  const [workAlias, setWorkAlias] = useState<string | null>(null);
  const [graphAlias, setGraphAlias] = useState<string | null>(null);

  const setWorkGraph = (work: string | null, graph: string | null) => {
    setWorkAlias(work);
    setGraphAlias(graph);
  };

  return (
    <FetcherTypeContext.Provider value={{ fetcherType, setFetcherType, workAlias, graphAlias, setWorkGraph }}>
      {children}
    </FetcherTypeContext.Provider>
  );
};
