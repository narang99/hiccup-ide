import { useState, useCallback } from 'react';
import { type Node } from '@xyflow/react';
import { type LayerNodeData } from '../components/nodes/LayerNode';

interface SelectedNode {
  id: string;
  data: LayerNodeData;
}

export const useSelectedNode = () => {
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);

  const handleNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (node.type === 'LayerNode') {
      setSelectedNode({ id: node.id, data: node.data as LayerNodeData });
    } else {
      setSelectedNode(null);
    }
  }, []);

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return {
    selectedNode,
    handleNodeClick,
    handlePaneClick,
    clearSelection,
  };
};