export interface POI {
  id?: number;
  work_alias: string;
  weight_coordinate: string;
  x: number;
  y: number;
  label: string;
  note: string;
}

export interface POIPoint {
  row: number;
  col: number;
  value: number;
}

export interface UniqueActivationId {
  input_alias: string;
  model_alias: string;
  coordinate: string;
}

export interface HighActivatedPOIOut {
  output_activation: UniqueActivationId;
  input_activations: UniqueActivationId[];
  point: POIPoint;
}

export interface HighActivatedPOIsResponse {
  pois: HighActivatedPOIOut[];
}

const apiBaseUrl = "http://localhost:8000";

export async function listPOIs(
  modelAlias: string, 
  inputAlias: string, 
  workAlias: string, 
  weightCoordinate: string
): Promise<POI[]> {
  try {
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/pois/${weightCoordinate}/`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to list POIs`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error listing POIs:", error);
    throw error;
  }
}

export async function savePOI(
  modelAlias: string, 
  inputAlias: string, 
  workAlias: string, 
  poi: POI
): Promise<POI> {
  try {
    const url = `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workAlias}/pois/`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(poi),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to save POI`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error saving POI:", error);
    throw error;
  }
}

export async function getHighActivatedPOIs(
  modelAlias: string,
  coordinate: string,
  k: number = 10
): Promise<HighActivatedPOIsResponse> {
  try {
    const url = `${apiBaseUrl}/api/models/${modelAlias}/coordinates/${coordinate}/high_pois/?k=${k}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to get high activated POIs`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error getting high activated POIs:", error);
    throw error;
  }
}
