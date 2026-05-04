export interface ActivationData {
  data: number[][] | number;
  shape: number[];
  layer_type: string;
  coordinate_type: string;
}

export async function loadActivationFromFile(
  coordinate: string,
  modelAlias: string,
  inputAlias: string,
  workAlias?: string,
  pruned?: boolean
): Promise<ActivationData> {
  try {
    const apiBaseUrl = "http://localhost:8000";

    const headers = { 'Content-Type': 'application/json' }
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/activations/single/${coordinate}/`;
    
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (pruned) params.append('pruned', 'true');
    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url, { headers, });
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