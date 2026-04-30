export interface ActivationData {
  data: number[][] | number;
  shape: number[];
  layer_type: string;
  coordinate_type: string;
}

export async function loadActivationFromFile(coordinate: string): Promise<ActivationData> {
  try {
    const headers = {'Content-Type': 'application/json'}
    const response = await fetch(`/activations/${coordinate}.json`, {headers,});
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load activation for coordinate: ${coordinate}`);
    }
    const data = await response.json();
    console.log(`Loaded activation for ${coordinate}:`, { shape: data.shape, dataType: typeof data.data });
    return data;
  } catch (error) {
    console.error(`Error loading activation for ${coordinate}:`, error);
    throw error;
  }
}