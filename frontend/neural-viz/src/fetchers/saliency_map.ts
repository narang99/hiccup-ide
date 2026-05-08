
export interface WorkGraphMeta {
  work_alias: string;
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
  pruned?: boolean
): Promise<LayerSaliencyData> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/layers/${layerName}/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (pruned) params.append('pruned', 'true');
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
  pruned?: boolean
): Promise<LayerSaliencyData> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/batch/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (pruned) params.append('pruned', 'true');
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
  modelAlias: string,
  inputAlias: string,
  workAlias?: string,
  pruned?: boolean
): Promise<LayerSaliencyMap> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    
    const headers = {'Content-Type': 'application/json'};
    let url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/single/${coordinate}/`;
    const params = new URLSearchParams();
    if (workAlias) params.append('work_alias', workAlias);
    if (pruned) params.append('pruned', 'true');
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

export async function markSliceDone(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  coordinate: string
): Promise<{ success: boolean; coordinate: string; is_done: boolean }> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/mark-done/`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ coordinate })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to mark slice as done for coordinate: ${coordinate}`);
    }
    
    const data = await response.json();
    console.log(`Marked slice as done for ${coordinate}`);
    return data;
  } catch (error) {
    console.error(`Error marking slice as done for ${coordinate}:`, error);
    throw error;
  }
}

export async function unmarkSliceDone(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  coordinate: string
): Promise<{ success: boolean; coordinate: string; is_done: boolean }> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/unmark-done/`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ coordinate })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to unmark slice as done for coordinate: ${coordinate}`);
    }
    
    const data = await response.json();
    console.log(`Unmarked slice as done for ${coordinate}`);
    return data;
  } catch (error) {
    console.error(`Error unmarking slice as done for ${coordinate}:`, error);
    throw error;
  }
}

export type SliceState = 'not_done' | 'skip' | 'review' | 'done';

export interface SliceStatus {
  coordinate: string;
  is_done: boolean;
  state: SliceState;
}

export async function getSliceStatus(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  coordinate: string
): Promise<SliceStatus> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/slice-status/${coordinate}/`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to get slice status for coordinate: ${coordinate}`);
    }
    
    const data = await response.json();
    // Handle backward compatibility - if state is not provided, infer from is_done
    if (!data.state) {
      data.state = data.is_done ? 'done' : 'not_done';
    }
    return data;
  } catch (error) {
    console.error(`Error getting slice status for ${coordinate}:`, error);
    throw error;
  }
}

export async function updateSliceState(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  coordinate: string,
  state: SliceState
): Promise<SliceStatus> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = {'Content-Type': 'application/json'};
    
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/update-slice-state/`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ coordinate, state })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to update slice state for coordinate: ${coordinate}`);
    }
    
    const data = await response.json();
    console.log(`Updated slice state to ${state} for ${coordinate}`);
    return data;
  } catch (error) {
    console.error(`Error updating slice state for ${coordinate}:`, error);
    throw error;
  }
}
