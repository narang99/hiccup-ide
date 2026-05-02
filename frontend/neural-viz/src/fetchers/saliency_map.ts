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

export interface LayerSaliencyMap {
  id: number;
  coordinate: string;
  data: number[][];
  shape: number[];
  coordinate_type: string;
  data_type: string;
}

export interface LayerSaliencyData {
  layer_name: string;
  saliency_maps: LayerSaliencyMap[];
}

export async function loadLayerSaliencyMaps(modelAlias: string, inputAlias: string, layerName: string): Promise<LayerSaliencyData> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    const response = await fetch(
      `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/layers/${layerName}/`,
      { headers }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load layer saliency maps for layer: ${layerName}`);
    }
    
    const data = await response.json();
    console.log(`Loaded layer saliency maps for ${layerName}:`, { 
      mapCount: data.saliency_maps?.length,
      layerName: data.layer_name 
    });
    
    return data;
  } catch (error) {
    console.error(`Error loading layer saliency maps for ${layerName}:`, error);
    throw error;
  }
}
