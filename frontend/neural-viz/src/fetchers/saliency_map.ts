import { type ActivationData } from './activation';

export async function loadSaliencyMapFromFile(coordinate: string): Promise<ActivationData> {
  try {
    const modelAlias = "example-model";
    const inputAlias = "first-input";
    const apiBaseUrl = "http://localhost:8000";
    
    const headers = {'Content-Type': 'application/json'};
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/${coordinate}/`, {headers,});
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
