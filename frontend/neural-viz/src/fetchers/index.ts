import { loadActivationFromFile } from './activation';
import { loadSaliencyMapFromFile } from './saliency_map';

export * from './activation';
export * from './saliency_map';

export type FetcherType = "activation" | "saliency_map";

// Define the main fetchers interface here since it combines multiple fetcher types
export interface NodeFetchers {
  activation?: (coordinate: string) => Promise<import('./activation').ActivationData>;
  saliency_map?: (coordinate: string) => Promise<import('./activation').ActivationData>;
}


// Create fetchers for data loading
export const DEFAULT_FETCHERS: NodeFetchers = {
  activation: loadActivationFromFile,
  saliency_map: loadSaliencyMapFromFile
};