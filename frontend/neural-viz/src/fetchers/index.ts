import { loadActivationFromFile } from './activation';
import { loadSaliencyMapFromFile, type LayerSaliencyMap } from './saliency_map';
import { loadWeightFromFile } from './weight';

export * from './activation';
export * from './saliency_map';
export * from './weight';

export type FetcherType = "activation" | "saliency_map" | "weight";

// Define the main fetchers interface here since it combines multiple fetcher types
export interface NodeFetchers {
  activation?: (coordinate: string, workAlias?: string, graphAlias?: string) => Promise<import('./activation').ActivationData>;
  saliency_map?: (coordinate: string, workAlias?: string, graphAlias?: string) => Promise<LayerSaliencyMap>;
  weight?: (coordinate: string, workAlias?: string, graphAlias?: string) => Promise<import('./weight').WeightData>;
}


// Create fetchers for data loading
export const DEFAULT_FETCHERS: NodeFetchers = {
  activation: loadActivationFromFile,
  saliency_map: loadSaliencyMapFromFile,
  weight: loadWeightFromFile
};