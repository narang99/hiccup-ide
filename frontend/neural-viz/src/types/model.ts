export interface ModelNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  shape: number[];
}

export interface ModelEdge {
  source: string;
  target: string;
}

export interface ModelData {
  nodes: ModelNode[];
  edges: ModelEdge[];
}

export interface ExpandedState {
  [nodeId: string]: boolean;
}

export interface KernelExpandedState {
  [kernelId: string]: boolean; // Format: "nodeId-kernel-index"
}