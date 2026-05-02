export interface NodeStats {
  min: number;
  max: number;
}

export async function fetchActivationsStats(
  modelAlias: string,
  inputAlias: string,
  coordinates: string[]
): Promise<NodeStats> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = { 'Content-Type': 'application/json' };
    const response = await fetch(
      `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/activations/stats/`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ coordinates })
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to fetch activations stats`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching activations stats:', error);
    throw error;
  }
}

export async function fetchSaliencyMapsStats(
  modelAlias: string,
  inputAlias: string,
  coordinates: string[]
): Promise<NodeStats> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const headers = { 'Content-Type': 'application/json' };
    const response = await fetch(
      `${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/saliency_maps/stats/`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ coordinates })
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to fetch saliency maps stats`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching saliency maps stats:', error);
    throw error;
  }
}
