import { type ActivationData } from './activation';

export async function loadSaliencyMapFromFile(coordinate: string): Promise<ActivationData> {
  try {
    const headers = {'Content-Type': 'application/json'};
    const response = await fetch(`/saliency_maps/${coordinate}.json`, {headers,});
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load saliency map for coordinate: ${coordinate}`);
    }
    const data = await response.json();
    console.log(`Loaded saliency map for ${coordinate}:`, { shape: data.shape, dataType: typeof data.data });
    return data;
  } catch (error) {
    console.error(`Error loading saliency map for ${coordinate}:`, error);
    throw error;
  }
}
