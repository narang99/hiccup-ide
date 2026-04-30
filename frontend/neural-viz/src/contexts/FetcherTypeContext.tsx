import { createContext, useContext, useState, type ReactNode } from 'react';
import { type FetcherType } from '../fetchers';

interface FetcherTypeContextValue {
  fetcherType: FetcherType;
  setFetcherType: (type: FetcherType) => void;
}

const FetcherTypeContext = createContext<FetcherTypeContextValue>({
  fetcherType: 'activation',
  setFetcherType: () => {},
});

export const FetcherTypeProvider = ({ children }: { children: ReactNode }) => {
  const [fetcherType, setFetcherType] = useState<FetcherType>('activation');
  return (
    <FetcherTypeContext.Provider value={{ fetcherType, setFetcherType }}>
      {children}
    </FetcherTypeContext.Provider>
  );
};

export const useFetcherType = () => useContext(FetcherTypeContext);
