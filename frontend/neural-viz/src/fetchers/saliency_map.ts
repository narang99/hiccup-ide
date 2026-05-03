
export interface WorkGraphMeta {
  work_alias: string;
  graph_alias: string;
}

export interface LayerSaliencyMap {
  id: number;
  coordinate: string;
  layer_name: string;
  data: number[][];
  shape: number[];
  coordinate_type: string;
  data_type: string;
  work_graph?: WorkGraphMeta | null;
}

export interface LayerSaliencyData {
  items: LayerSaliencyMap[];
}

export async function loadLayerSaliencyMaps(
  modelAlias: string, 
  inputAlias: string, 
  layerName: string,
  workAlias?: string,
  graphAlias?: string
): Promise<LayerSaliencyData> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/layers/${layerName}/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (graphAlias) params.append('graph_alias', graphAlias);
    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url, { headers });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load layer saliency maps for layer: ${layerName}`);
    }
    
    const data = await response.json();
    console.log(`Loaded layer saliency maps for ${layerName}:`, { 
      mapCount: data.items?.length,
    });
    
    return data;
  } catch (error) {
    console.error(`Error loading layer saliency maps for ${layerName}:`, error);
    throw error;
  }
}

export async function loadBatchSaliencyMaps(
  modelAlias: string, 
  inputAlias: string, 
  coordinates: string[],
  workAlias?: string,
  graphAlias?: string
): Promise<LayerSaliencyData> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/batch/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (graphAlias) params.append('graph_alias', graphAlias);
    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url, { 
      method: 'POST',
      headers,
      body: JSON.stringify({ coordinates })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load batch saliency maps`);
    }
    
    const data = await response.json();
    console.log(`Loaded batch saliency maps:`, { 
      mapCount: data.items?.length,
      requestedCount: coordinates.length
    });
    
    return data;
  } catch (error) {
    console.error(`Error loading batch saliency maps:`, error);
    throw error;
  }
}

export async function loadSaliencyMapFromFile(
  coordinate: string,
  workAlias?: string,
  graphAlias?: string
): Promise<LayerSaliencyMap> {
  try {
    const modelAlias = "example-model";
    const inputAlias = "first-input";
    const apiBaseUrl = "http://localhost:8000";
    
    const headers = {'Content-Type': 'application/json'};
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/single/${coordinate}/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (graphAlias) params.append('graph_alias', graphAlias);
    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url, {headers,});
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
