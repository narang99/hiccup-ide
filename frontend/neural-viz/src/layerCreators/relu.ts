import { type Node, type XYPosition } from '@xyflow/react';
import { DEFAULT_FETCHERS, type FetcherType } from "../fetchers";
import type { ModelNode } from "../types/model";
import { makeLayerLayout } from "./layout";
import { ReluChannelNodeData } from '../components/nodes/ReluChannelNode';

export const createReLULayer = (
    modelNode: ModelNode,
    basePosition: { x: number; y: number },
    fetcherType: FetcherType
): Node[] => {
    const nodes: Node[] = [];
    // Get number of channels from the shape (assuming format [batch, channels, height, width])
    const numChannels = modelNode.shape.length > 1 ? modelNode.shape[1] : 1;
    const childWidth = 120;
    const childHeight = 120;
    const padding = 10;
    const layout = makeLayerLayout(numChannels, childHeight, childWidth, padding);

    // Create parent layer node
    nodes.push(
        makeParentLayerNode(modelNode.id, basePosition, layout.parent.width, layout.parent.height, numChannels)
    );

    // Create channel sub-nodes with parentId in grid layout
    for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
        const childPosition = layout.children[channelIndex];
        nodes.push(makeReluChannelNode(
            modelNode.id, childPosition, childHeight, childWidth, channelIndex, fetcherType
        ))
    }

    return nodes;
};


const makeParentLayerNode = (modelId: string, position: XYPosition, width: number, height: number, numChannels: number): Node => {
    return {
        id: modelId,
        type: 'LayerNode',
        position: position,
        width: width,
        height: height,
        data: {
            label: `ReLU Layer (${numChannels} channels)`,
            layerType: 'ReLU' as const,
            nodeCount: numChannels,
        },
    };
}


const makeReluChannelNode = (
    modelId: string, position: XYPosition, height: number, width: number, channelIndex: number, fetcherType: FetcherType
): Node => {
    return {
        id: `${modelId}-channel-${channelIndex}`,
        type: 'default',
        position: position,
        data: {
            label: ReluChannelNodeData({ channelIndex, layerId: modelId, fetchers: DEFAULT_FETCHERS, fetcherType }),
        },
        width: width,
        height: height,
        style: {
            background: 'transparent',
            border: '1px solid rgba(251, 191, 36, 0.35)',
            borderRadius: '6px',
            padding: 0,
            fontSize: '10px',
            overflow: 'hidden',
        },
        parentId: modelId,
        extent: 'parent',
    };

}