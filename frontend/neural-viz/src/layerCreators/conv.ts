import { type Node, type XYPosition } from '@xyflow/react';
import { DEFAULT_FETCHERS, type FetcherType } from "../fetchers";
import type { ModelNode } from "../types/model";
import { createOutputKernelNode } from '../utils/kernelNodes';
import { makeEvenlySpacedHorizontalLayout } from '../layouts/horizontal';
import { type HandleDirection } from '../components/nodes/ActivationFlowNode';
import type { LayerGroupLayout } from '../layouts/common';

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
    const layout = makeEvenlySpacedHorizontalLayout(outChannels, childHeight, childWidth, padding);
    // inside evenly spaced layout, we dont make edges so dont need handles
    const handleDirection: HandleDirection = null;

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
            fetcherType,
            handleDirection,
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