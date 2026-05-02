import { type NodeProps, type Node } from '@xyflow/react';
import type { NodeFetchers, FetcherType } from "../../fetchers";
import BaseActivationNode from "./BaseActivationNode";

export interface ActivationNodeData extends Record<string, unknown> {
    coordinate: string;
    fetchers?: NodeFetchers;
    fetcherType?: FetcherType;
    maxSize?: number;
    title: string;
    badgeLabel?: string;
    badgeColor?: string;
}

export type ActivationNodeType = Node<ActivationNodeData, 'ActivationNode'>;

export const ActivationFlowNode = ({ data }: NodeProps) => {
    const typedData = data as ActivationNodeData;
    
    return (
        <BaseActivationNode {...typedData} />
    );
};