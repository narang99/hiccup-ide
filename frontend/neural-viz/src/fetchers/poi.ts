export interface POI {
  id?: number;
  work_alias: string;
  weight_coordinate: string;
  x: number;
  y: number;
  label: string;
  note: string;
}

const apiBaseUrl = "http://localhost:8000";

export async function listPOIs(workAlias: string, weightCoordinate: string): Promise<POI[]> {
  try {
    const url = `${apiBaseUrl}/api/poi/${workAlias}/${weightCoordinate}/`;
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

export async function savePOI(poi: POI): Promise<POI> {
  try {
    const url = `${apiBaseUrl}/api/poi/`;
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
