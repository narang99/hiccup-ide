export interface LayerThreshold {
  id: number;
  layer_id: string;
  slider_value: number;
}

export interface SaveThresholdRequest {
  layer_id: string;
  slider_value: number;
}

const API_BASE_URL = "http://localhost:8000";

export async function loadWorkflowThresholds(
  modelAlias: string, 
  inputAlias: string, 
  workflowName: string
): Promise<LayerThreshold[]> {
  try {
    const headers = {'Content-Type': 'application/json'};
    
    const response = await fetch(
      `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/thresholds/`,
      { headers }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load workflow thresholds`);
    }
    
    const data = await response.json();
    console.log(`Loaded workflow thresholds for ${workflowName}:`, data);
    
    return data;
  } catch (error) {
    console.error(`Error loading workflow thresholds for ${workflowName}:`, error);
    throw error;
  }
}

export async function saveWorkflowThreshold(
  modelAlias: string,
  inputAlias: string,
  workflowName: string,
  threshold: SaveThresholdRequest
): Promise<LayerThreshold> {
  try {
    const headers = {'Content-Type': 'application/json'};
    
    const response = await fetch(
      `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/thresholds/`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(threshold)
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to save threshold`);
    }
    
    const data = await response.json();
    console.log(`Saved threshold for layer ${threshold.layer_id}:`, data);
    
    return data;
  } catch (error) {
    console.error(`Error saving threshold:`, error);
    throw error;
  }
}