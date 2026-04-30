export * from './activation';

// Define the main fetchers interface here since it combines multiple fetcher types
export interface NodeFetchers {
  activation: (coordinate: string) => Promise<import('./activation').ActivationData>;
  // Future: saliency, gradients, etc.
}