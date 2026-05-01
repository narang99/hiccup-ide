import type { ModelNode } from "../types/model";
import { type Node } from '@xyflow/react';

export const createOtherLayer = (
    modelNode: ModelNode,
    basePosition: { x: number; y: number }
): Node[] => {
    const layerNode = {
        id: modelNode.id,
        type: 'LayerNode',
        position: basePosition,
        data: {
            label: modelNode.type,
            layerType: modelNode.type as 'Linear' | 'Flatten' | 'Input' | 'Output',
        },
        width: 150,
        height: 100,
    };

    return [layerNode];
};