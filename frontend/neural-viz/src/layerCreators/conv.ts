import { type Node, type XYPosition } from '@xyflow/react';
import { DEFAULT_FETCHERS, type FetcherType } from "../fetchers";
import type { ModelNode } from "../types/model";
import { makeLayerLayout, type LayerGroupLayout } from "./layout";
import { createOutputKernelNode } from '../utils/kernelNodes';

export const createConv2dLayer = (
    modelNode: ModelNode,
    basePosition: { x: number; y: number },
    fetcherType: FetcherType
): Node[] => {
    const nodes: Node[] = [];
    const outChannels = modelNode.params.out_channels as number;
    const childWidth = 130;
    const childHeight = 150;
    const padding = 10;
    const layout = makeLayerLayout(outChannels, childHeight, childWidth, padding);

    nodes.push(makeParentNode(modelNode.id, layout, basePosition, outChannels));

    for (let kernelIndex = 0; kernelIndex < outChannels; kernelIndex++) {
        const childPosition = layout.children[kernelIndex];
        nodes.push(createOutputKernelNode(
            modelNode,
            kernelIndex,
            childHeight,
            childWidth,
            childPosition,
            DEFAULT_FETCHERS,
            fetcherType
        ))
    }
    return nodes;
};


const makeParentNode = (modelId: string, layout: LayerGroupLayout, basePosition: XYPosition, outChannels: number): Node => {
    const parentLayerNode: Node = {
        id: modelId,
        type: 'LayerNode',
        position: basePosition,
        data: {
            label: `Conv2d Layer (${outChannels} kernels)`,
            layerType: 'Conv2d' as const,
            nodeCount: outChannels,
        },
        width: layout.parent.width,
        height: layout.parent.height,
    };
    return parentLayerNode;
}