import { create } from 'zustand';
import { type Node } from '@xyflow/react';
import { type LayerNodeData } from '../components/nodes/LayerNode';
import { useLayerSaliencyStore } from './layerSaliencyStore';

interface SelectedNode {
  id: string;
  data: LayerNodeData;
}

interface SelectedNodeState {
  selectedNode: SelectedNode | null;
  setSelectedNode: (node: SelectedNode | null) => void;
  handleNodeClick: (event: React.MouseEvent, node: Node) => void;
  handlePaneClick: () => void;
  clearSelection: () => void;
}

export const useSelectedNodeStore = create<SelectedNodeState>((set) => ({
  selectedNode: null,
  
  setSelectedNode: (node: SelectedNode | null) => {
    // Clear saliency cache when selected node changes
    useLayerSaliencyStore.getState().clearCache();
    set({ selectedNode: node });
  },
  
  handleNodeClick: (_event: React.MouseEvent, node: Node) => {
    if (node.type === 'LayerNode') {
      // Clear saliency cache when selecting a new node
      useLayerSaliencyStore.getState().clearCache();
      set({ selectedNode: { id: node.id, data: node.data as LayerNodeData } });
    } else {
      // Clear saliency cache when deselecting
      useLayerSaliencyStore.getState().clearCache();
      set({ selectedNode: null });
    }
  },
  
  handlePaneClick: () => {
    // Clear saliency cache when clicking pane
    useLayerSaliencyStore.getState().clearCache();
    set({ selectedNode: null });
  },
  
  clearSelection: () => {
    // Clear saliency cache when clearing selection
    useLayerSaliencyStore.getState().clearCache();
    set({ selectedNode: null });
  },
}));