import { type NodeProps, type Node } from '@xyflow/react';
import type { NodeFetchers, FetcherType } from "../../fetchers";
import BaseActivationNode from "./BaseActivationNode";
import type { ActivationFilterAlgorithm } from '../../types/activationFiltering';
import type { OverlayAlgorithm } from '../../types/overlay';

export type HandleDirection = "TB" | "LR" | null;

export type ClickAction = 
    | { type: 'link'; link: string }
    | { type: 'callback'; callback: (nodeId: string, coordinate: string, gridCoord: [number, number] | null, position: [number, number] | null) => void };

export interface ActivationNodeData extends Record<string, unknown> {
    coordinate: string;
    fetchers?: NodeFetchers;
    fetcherType?: FetcherType;
    maxSize?: number;
    title: string;
    badgeLabel?: string;
    badgeColor?: string;
    handleDirection?: HandleDirection;
    clickAction?: ClickAction;
    filterAlgorithm?: ActivationFilterAlgorithm;
    absMax?: number;
    onPixelHover?: (nodeId: string, coordinate: string, gridCoord: [number, number], position: [number, number]) => void;
    onPixelLeave?: (nodeId: string, coordinate: string) => void;
    overlayAlgorithm?: OverlayAlgorithm;
    modelAlias: string;
    inputAlias: string;
    workAlias?: string;
}

export type ActivationNodeType = Node<ActivationNodeData, 'ActivationNode'>;

export const ActivationFlowNode = ({ data }: NodeProps) => {
    const typedData = data as ActivationNodeData;
    
    return (
        <BaseActivationNode 
            {...typedData} 
            handleDirection={typedData.handleDirection} 
            clickAction={typedData.clickAction} 
            filterAlgorithm={typedData.filterAlgorithm}
            absMax={typedData.absMax}
            modelAlias={typedData.modelAlias}
            inputAlias={typedData.inputAlias}
            workAlias={typedData.workAlias}
        />
    );
};