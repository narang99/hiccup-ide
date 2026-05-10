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
    const layout = makeEvenlySpacedLayout(1, childHeight, childWidth, childPadding, childDirection);
    nodes.push(getLayerNode(layerNodeId, layout.parent.width, layout.parent.height, 1, childDirection));

    const parentId = layerNodeId;
    const position = layout.children[0]
    const rect: OverlayAlgorithm = {
        type: 'DrawRect',
        start: [uiNode.input_patch.patch_min_x, uiNode.input_patch.patch_min_y],
        end: [uiNode.input_patch.patch_max_x + 1, uiNode.input_patch.patch_max_y + 1],
    };

    nodes.push(createActivationNode(
        nodeId, 
        "Conv2dOp", 
        uiNode, 
        modelAlias, 
        inputAlias, 
        workAlias, 
        rect, 
        fetcherType, 
        position, 
        parentId, 
        childHeight, 
        childWidth,
    ));

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
        start: [uiNode.x, uiNode.y],
        end: [uiNode.x + 1, uiNode.y + 1],
    };

    nodes.push(createActivationNode(
        nodeId, 
        "Conv2dOutputCoord", 
        uiNode, 
        modelAlias, 
        inputAlias, 
        workAlias, 
        rect, 
        fetcherType, 
        position, 
        parentId, 
        childHeight, 
        childWidth,
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