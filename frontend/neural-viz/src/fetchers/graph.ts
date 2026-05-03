const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface WorkGraphResponse {
  id: number;
  alias: string;
  created: boolean;
}

export async function createOrUpdateWorkGraph(
  modelAlias: string,
  inputAlias: string,
  workflowName: string,
  graphAlias: string
): Promise<WorkGraphResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/graphs/${graphAlias}/`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to create or update work graph');
  }

  return response.json();
}

export interface PruningStatusResponse {
  layers: {
    done: string[];
    total: string[];
  };
}

export async function getPruningStatus(
  modelAlias: string,
  inputAlias: string,
  workflowName: string,
  graphAlias: string
): Promise<PruningStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/graphs/${graphAlias}/status/`,
    {
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to fetch pruning status');
  }

  return response.json();
}
