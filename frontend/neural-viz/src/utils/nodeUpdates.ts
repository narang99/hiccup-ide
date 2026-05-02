import { type Node } from '@xyflow/react';

/**
 * Updates absMax values for ActivationFlowNode nodes while preserving selection state.
 * Only creates new node objects if the absMax value actually changed.
 * 
 * @param nodes - Current nodes array
 * @param layerAbsMax - Map of layer ID to absMax value
 * @returns Updated nodes array (same reference if no changes)
 */
export const updateNodeAbsMax = (
  nodes: Node[],
  layerAbsMax: Record<string, number>
): Node[] => {
  let hasChanges = false;
  
  const updatedNodes = nodes.map((node) => {
    if (node.type === 'ActivationFlowNode' && node.parentId && layerAbsMax[node.parentId] !== undefined) {
      // Only update if absMax actually changed
      if (node.data.absMax !== layerAbsMax[node.parentId]) {
        hasChanges = true;
        return {
          ...node,
          data: {
            ...node.data,
            absMax: layerAbsMax[node.parentId]
          }
        };
      }
    }
    return node;
  });
  
  // Only return new array if there were actual changes
  return hasChanges ? updatedNodes : nodes;
};