import type { ActivationFilterAlgorithm } from '../types/activationFiltering';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface WorkGraphResponse {
  id: number;
  created: boolean;
}

export async function createOrUpdateWorkGraph(
  modelAlias: string,
  inputAlias: string,
  workflowName: string
): Promise<WorkGraphResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/`,
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
  session_active: boolean;
}

export async function getPruningStatus(
  modelAlias: string,
  inputAlias: string,
  workflowName: string
): Promise<PruningStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/status/`,
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

export interface CoordinateAlgorithm {
  coordinate: string;
  algorithm: ActivationFilterAlgorithm;
}

export interface BatchWorkSaliencyMapsResponse {
  created: number;
  updated: number;
}

export async function saveWorkSaliencyMaps(
  modelAlias: string,
  inputAlias: string,
  workflowName: string,
  items: CoordinateAlgorithm[]
): Promise<BatchWorkSaliencyMapsResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/saliency_maps/`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ items }),
    }
  );

  if (!response.ok) {
    throw new Error('Failed to save work saliency maps');
  }

  return response.json();
}

export async function startPruning(
  modelAlias: string,
  inputAlias: string,
  workflowName: string
): Promise<{ status: string; cloned_count: number }> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/start_pruning/`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to start pruning session');
  }

  return response.json();
}

export async function finalizePruning(
  modelAlias: string,
  inputAlias: string,
  workflowName: string
): Promise<{ status: string; committed_count: number }> {
  const response = await fetch(
    `${API_BASE_URL}/api/models/${modelAlias}/inputs/${inputAlias}/workflows/${workflowName}/finalize_pruning/`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to finalize pruning session');
  }

  return response.json();
}
