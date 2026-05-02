import { type NodeProps, type Node } from '@xyflow/react';
import type { NodeFetchers, FetcherType } from "../../fetchers";
import BaseActivationNode from "./BaseActivationNode";

export type HandleDirection = "TB" | "LR" | null;

export interface ActivationNodeData extends Record<string, unknown> {
    coordinate: string;
    fetchers?: NodeFetchers;
    fetcherType?: FetcherType;
    maxSize?: number;
    title: string;
    badgeLabel?: string;
    badgeColor?: string;
    handleDirection?: HandleDirection;
    link?: string;
}

export type ActivationNodeType = Node<ActivationNodeData, 'ActivationNode'>;

export const ActivationFlowNode = ({ data }: NodeProps) => {
    const typedData = data as ActivationNodeData;
    
    return (
        <BaseActivationNode {...typedData} handleDirection={typedData.handleDirection} link={typedData.link} />
    );
};