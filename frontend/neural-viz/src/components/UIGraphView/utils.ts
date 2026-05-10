import '@xyflow/react/dist/style.css';
import { type Node, type XYPosition } from '@xyflow/react';
import { DEFAULT_FETCHERS, type FetcherType } from '../../fetchers';
import type { OverlayAlgorithm } from '../../types/overlay';
import { toggleDirection, type Direction } from '../../types/direction';

// export function getCoordinateString(coord: Coordinate): string {
//     switch (coord.type) {
//         case 'Conv2dOutputCoordinate':
//             return `${coord.layer_name}.out_${coord.channel}`
//         case 'SingleConv2dOpNode':
//             return `${coord.input_patch.layer_name}.out_${coord.input_patch.channel}`
//         default:
//             // Fallback for types that might not have channel explicitly but are still activation-like
//             if ('layer_name' in coord) {
//                 return `${coord.layer_name}.out_0`;
//             }
//             throw new Error(`Coordinate string generation not implemented for ${JSON.stringify(coord)}`);
//     }
// }

export function createActivationNode(
    nodeId: string,
    label: string,
    coordinate: string,
    modelAlias: string,
    inputAlias: string,
    workAlias: string | undefined,
    fetcherType: FetcherType,
    position: XYPosition,
    parentId: string,
    height: number,
    width: number,
    overlay?: OverlayAlgorithm,
): Node {
    return {
        id: nodeId,
        type: 'ActivationFlowNode',
        height,
        width,
        data: {
            handleDirection: null,
            coordinate,
            modelAlias,
            inputAlias,
            workAlias,
            title: label,
            fetchers: DEFAULT_FETCHERS,
            fetcherType: fetcherType,
            overlayAlgorithm: overlay,
        },
        position,
        parentId,
        extent: 'parent',
    };
}

/**
 * Creates a default React Flow node.
 */
export function createDefaultNode(nodeId: string, label: string, position: XYPosition, parentId: string, height: number, width: number): Node {
    return {
        id: nodeId,
        type: 'default',
        data: { label },
        height,
        width,
        position,
        parentId,
        extent: 'parent',
    };
}


export const getLayerNode = (
    id: string, width: number, height: number, nodeCount: number, pageDirection: Direction
) => {
    return {
        id: id,
        type: 'LayerNode',
        position: { x: 0, y: 0 }, // empty position, filled by dagre
        width: width,
        height: height,
        data: {
            label: "",
            layerType: '',
            nodeCount: nodeCount,
            handleDirection: toggleDirection(pageDirection),
        },
    };
}

