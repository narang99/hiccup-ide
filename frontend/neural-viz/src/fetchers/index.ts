import { loadActivationFromFile } from './activation';

export * from './activation';

// Define the main fetchers interface here since it combines multiple fetcher types
export interface NodeFetchers {
  activation?: (coordinate: string) => Promise<import('./activation').ActivationData>;
}


// Create fetchers for data loading
export const DEFAULT_FETCHERS: NodeFetchers = {
  activation: loadActivationFromFile
};