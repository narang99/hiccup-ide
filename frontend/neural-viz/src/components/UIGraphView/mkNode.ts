import type { Node } from "@xyflow/react";
import type { FetcherType } from "../../fetchers";
import { makeEvenlySpacedLayout } from "../../layouts";
import type { Conv2dOutputCoordinate, Coordinate, SingleConv2dOpNode } from "../../types/coordinates";
import type { Direction } from "../../types/direction";
import type { OverlayAlgorithm } from "../../types/overlay";
import { createActivationNode, createDefaultNode, getLayerNode } from "./utils";

export const makeConv2dOpNodes = (
    nodeId: string,
    uiNode: SingleConv2dOpNode,
    modelAlias: string,
    inputAlias: string,
    workAlias: string,
    fetcherType: FetcherType,
    childHeight: number,
    childWidth: number,
    childPadding: number,
    childDirection: Direction,
): [string, Node[]] => {
    const nodes: Node[] = [];
    const layerNodeId = `layer-${nodeId}`;
    const parentId = layerNodeId;

    // layout
    const layout = makeEvenlySpacedLayout(3, childHeight, childWidth, childPadding, childDirection);

    // push layer node
    nodes.push(getLayerNode(layerNodeId, layout.parent.width, layout.parent.height, 3, childDirection));


    // push input activation with rect
    const rect: OverlayAlgorithm = {
        type: 'DrawRect',
        rects: [{
            start: [uiNode.input_patch.patch_min_x, uiNode.input_patch.patch_min_y],
            end: [uiNode.input_patch.patch_max_x + 1, uiNode.input_patch.patch_max_y + 1],
        }],
    };
    const sliceLink = `/models/${modelAlias}/${inputAlias}/${workAlias}/kernel-slice/${uiNode.output_slice.layer_name}/${uiNode.output_slice.out_channel}/${uiNode.output_slice.in_channel}`;

    nodes.push(createActivationNode(
        `${nodeId}-input`, 
        "Conv2dOp", 
        `${uiNode.input_patch.layer_name}.out_${uiNode.input_patch.channel}`, 
        modelAlias, 
        inputAlias, 
        workAlias, 
        fetcherType, 
        layout.children[0], 
        parentId, 
        childHeight, 
        childWidth,
        rect, 
        sliceLink,
    ));

    // push weight
    nodes.push(createActivationNode(
        `${nodeId}-weight`, 
        "Conv2dOp", 
        `${uiNode.output_slice.layer_name}.out_${uiNode.output_slice.out_channel}.in_${uiNode.output_slice.in_channel}`, 
        modelAlias, 
        inputAlias, 
        workAlias, 
        "weight", 
        layout.children[1], 
        parentId,
        childHeight, 
        childWidth,
        undefined,
        sliceLink,
    ));
    
    // push output
    nodes.push(createActivationNode(
        `${nodeId}-output`, 
        "Conv2dOp", 
        `${uiNode.output_slice.layer_name}.out_${uiNode.output_slice.out_channel}.in_${uiNode.output_slice.in_channel}`, 
        modelAlias, 
        inputAlias, 
        workAlias, 
        fetcherType, 
        layout.children[2], 
        parentId,
        childHeight, 
        childWidth,
        undefined,
        sliceLink,
    ));
    // push kernel

    return [layerNodeId, nodes]
}


export const makeConv2dOutputCoordNodes = (
    nodeId: string,
    uiNode: Conv2dOutputCoordinate,
    modelAlias: string,
    inputAlias: string,
    workAlias: string,
    fetcherType: FetcherType,
    childHeight: number,
    childWidth: number,
    childPadding: number,
    childDirection: Direction,
): [string, Node[]] => {
    const nodes: Node[] = [];
    const layerNodeId = `layer-${nodeId}`;
    const layout = makeEvenlySpacedLayout(1, childHeight, childWidth, childPadding, childDirection);
    nodes.push(getLayerNode(layerNodeId, layout.parent.width, layout.parent.height, 1, childDirection));

    const parentId = layerNodeId;
    const position = layout.children[0]
    const rect: OverlayAlgorithm = {
        type: 'DrawRect',
        rects: [{
            start: [uiNode.x, uiNode.y],
            end: [uiNode.x + 1, uiNode.y + 1],
        }],
    };

    const contribsLink = `/models/${modelAlias}/${inputAlias}/${workAlias}/kernel/${uiNode.layer_name}/${uiNode.channel}`;

    nodes.push(createActivationNode(
        nodeId, 
        "Conv2dOutputCoord", 
        `${uiNode.layer_name}.out_${uiNode.channel}`,
        modelAlias, 
        inputAlias, 
        workAlias, 
        fetcherType, 
        position, 
        parentId, 
        childHeight, 
        childWidth,
        rect, 
        contribsLink,
    ));

    return [layerNodeId, nodes]
}


export const makeDefaultNodes = (
    nodeId: string,
    uiNode: Coordinate,
    childHeight: number,
    childWidth: number,
    childPadding: number,
    childDirection: Direction,
): [string, Node[]] => {
    const nodes: Node[] = [];
    const layerNodeId = `layer-${nodeId}`;
    const layout = makeEvenlySpacedLayout(1, childHeight, childWidth, childPadding, childDirection);
    nodes.push(getLayerNode(layerNodeId, layout.parent.width, layout.parent.height, 1, childDirection));

    const parentId = layerNodeId;
    const position = layout.children[0]

    const label = `${uiNode.layer_name} / ${uiNode.type}`;
    nodes.push(createDefaultNode(nodeId, label, position, parentId, childHeight, childWidth));

    return [layerNodeId, nodes]
}