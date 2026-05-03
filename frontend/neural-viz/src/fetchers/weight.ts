export interface WeightData {
  data: number[][] | number;
  shape: number[];
  layer_type: string;
  coordinate_type: string;
  data_type: string;
}

export async function loadWeightFromFile(
  coordinate: string,
  workAlias?: string,
  graphAlias?: string
): Promise<WeightData> {
  try {
    const modelAlias = "example-model";
    const apiBaseUrl = "http://localhost:8000";

    const headers = { 'Content-Type': 'application/json' };
    let url = `${apiBaseUrl}/api/models/${modelAlias}/weights/single/${coordinate}/`;
    
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (graphAlias) params.append('graph_alias', graphAlias);
    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url, { headers });
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
