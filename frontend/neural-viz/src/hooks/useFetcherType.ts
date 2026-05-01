import { useContext } from 'react';
import { FetcherTypeContext } from '../contexts/FetcherTypeContextDefinition';

export const useFetcherType = () => useContext(FetcherTypeContext);