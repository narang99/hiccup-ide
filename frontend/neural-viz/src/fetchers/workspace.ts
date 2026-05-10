export interface WorkTree {
  alias: string;
  name: string;
  is_pinned: boolean;
  has_graph: boolean;
}

export interface InputTree {
  alias: string;
  name: string;
  works: WorkTree[];
}

export interface ModelTree {
  alias: string;
  name: string;
  inputs: InputTree[];
}

export async function fetchWorkspace(): Promise<ModelTree[]> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/workspace/`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to fetch workspace hierarchy`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching workspace:", error);
    throw error;
  }
}

export async function createWork(modelAlias: string, inputAlias: string, name: string): Promise<WorkTree> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/works/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to create work`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error creating work:", error);
    throw error;
  }
}

export async function pinWork(modelAlias: string, inputAlias: string, workAlias: string): Promise<{success: boolean, action: string}> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/works/${workAlias}/pin/`, {
      method: 'POST'
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to pin work`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error pinning work:", error);
    throw error;
  }
}

export async function unpinWork(modelAlias: string, inputAlias: string, workAlias: string): Promise<{success: boolean, action: string}> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/inputs/${inputAlias}/works/${workAlias}/pin/`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to unpin work`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error unpinning work:", error);
    throw error;
  }
}

export async function getPinnedWorksForModel(modelAlias: string): Promise<WorkTree[]> {
  try {
    const apiBaseUrl = "http://localhost:8000";
    const response = await fetch(`${apiBaseUrl}/api/models/${modelAlias}/pinned-works/`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to fetch pinned works`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching pinned works:", error);
    throw error;
  }
}
