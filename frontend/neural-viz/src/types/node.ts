import type { Direction } from "./direction";

export interface SelectedNode {
  id: string;
  data: LayerNodeData;
}

export interface LayerNodeData extends Record<string, unknown> {
  label: string;
  layerType: 'Conv2d' | 'ReLU' | 'Linear' | 'Flatten' | 'Input' | 'Output';
  nodeCount?: number;
  handleDirection: Direction | null;
}
