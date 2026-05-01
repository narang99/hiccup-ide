export interface WeightData {
  data: number[][] | number;
  shape: number[];
  layer_type: string;
  coordinate_type: string;
  data_type: string;
}

export async function loadWeightFromFile(coordinate: string): Promise<WeightData> {
  try {
    const modelAlias = "example-model";
    const apiBaseUrl = "http://localhost:8000";

    const headers = { 'Content-Type': 'application/json' };
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/weights/${coordinate}/`, { headers });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load weight for coordinate: ${coordinate}`);
    }
    const data = await response.json();
    console.log(`Loaded weight for ${coordinate}:`, { shape: data.shape, dataType: typeof data.data });
    return data;
  } catch (error) {
    console.error(`Error loading weight for ${coordinate}:`, error);
    throw error;
  }
}
