import { type Node, type XYPosition } from '@xyflow/react';
import { DEFAULT_FETCHERS, type FetcherType } from "../fetchers";
import type { ModelNode } from "../types/model";
import { createOutputKernelNode } from '../utils/kernelNodes';
import { type HandleDirection } from '../components/nodes/ActivationFlowNode';
import type { LayerGroupLayout } from '../layouts/common';
import type { Direction } from '../types/direction';
import { makeEvenlySpacedLayout } from '../layouts';

export const createConv2dLayer = (
    modelNode: ModelNode,
    basePosition: { x: number; y: number },
    fetcherType: FetcherType,
    layerBlockHandleDirection: Direction,
    directionInsideLayerBlock: Direction = "LR",
): Node[] => {
    const nodes: Node[] = [];
    const outChannels = modelNode.params.out_channels as number;
    const childWidth = 130;
    const childHeight = 150;
    const padding = 10;
    const layout = makeEvenlySpacedLayout(outChannels, childHeight, childWidth, padding, directionInsideLayerBlock);
    // inside evenly spaced layout, we dont make edges so dont need handles
    const handleDirection: HandleDirection = null;
    // parent handle should have the same direction as the page

    nodes.push(makeParentNode(modelNode.id, layout, basePosition, outChannels, layerBlockHandleDirection));

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


const makeParentNode = (modelId: string, layout: LayerGroupLayout, basePosition: XYPosition, outChannels: number, handleDirection: null | Direction): Node => {
    const parentLayerNode: Node = {
        id: modelId,
        type: 'LayerNode',
        position: basePosition,
        data: {
            label: `Conv2d Layer (${outChannels} kernels)`,
            layerType: 'Conv2d' as const,
            nodeCount: outChannels,
            handleDirection: handleDirection,
        },
        width: layout.parent.width,
        height: layout.parent.height,
    };
    return parentLayerNode;
}