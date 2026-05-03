import { create } from 'zustand';
import { type Node } from '@xyflow/react';
import type { LayerNodeData, SelectedNode } from '../types/node';

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
    set({ selectedNode: node });
  },
  
  handleNodeClick: (_event: React.MouseEvent, node: Node) => {
    if (node.type === 'LayerNode') {
      set({ selectedNode: { id: node.id, data: node.data as LayerNodeData } });
    } else {
      set({ selectedNode: null });
    }
  },
  
  handlePaneClick: () => {
    set({ selectedNode: null });
  },
  
  clearSelection: () => {
    set({ selectedNode: null });
  },
}));