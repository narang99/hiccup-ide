import { useState, type ReactNode } from 'react';
import { type FetcherType } from '../fetchers';
import { FetcherTypeContext } from './FetcherTypeContextDefinition';

export const FetcherTypeProvider = ({ children }: { children: ReactNode }) => {
  const [fetcherType, setFetcherType] = useState<FetcherType>('activation');
  return (
    <FetcherTypeContext.Provider value={{ fetcherType, setFetcherType }}>
      {children}
    </FetcherTypeContext.Provider>
  );
};
