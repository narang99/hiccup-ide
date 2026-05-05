export interface WorkTree {
  alias: string;
  name: string;
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
