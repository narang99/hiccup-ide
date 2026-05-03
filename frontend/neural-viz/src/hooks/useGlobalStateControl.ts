import { useEffect, useState } from 'react';
import { type Node } from '@xyflow/react';
import { type FetcherType } from '../fetchers';
import { updateNodeAbsMax } from '../utils/nodeUpdates';
import { useLayerStats } from './useLayerStats';
import { useColormap } from './useColormap';

interface UseGlobalStateControlProps {
    nodes: Node[];
    fetcherType: FetcherType;
    setNodes: (value: React.SetStateAction<Node[]>) => void
    modelAlias?: string;
    inputAlias?: string;
}

export const useGlobalStateControl = ({
    nodes,
    fetcherType,
    setNodes,
    modelAlias = "example-model",
    inputAlias = "first-input"
}: UseGlobalStateControlProps) => {
    // tracks the scaling mode and scales the nodes automatically
    // make sure setNodes is simply the function you use for setting the nodes state of react flow
    const [layerAbsMax, setLayerAbsMax] = useState<Record<string, number>>({});
    const { scalingMode } = useColormap();
    useLayerStats({ nodes, fetcherType, scalingMode, setLayerAbsMax, modelAlias, inputAlias })
    useEffect(() => {
        if (scalingMode === 'global' && Object.keys(layerAbsMax).length > 0) {
            setNodes((currentNodes) => updateNodeAbsMax(currentNodes, layerAbsMax));
        }
    }, [layerAbsMax, scalingMode, setNodes]);

    useEffect(() => {
        if (scalingMode === 'global' && Object.keys(layerAbsMax).length > 0) {
            if (setNodes) {
                setNodes((currentNodes) => updateNodeAbsMax(currentNodes, layerAbsMax));
            }
        }
    }, [layerAbsMax, scalingMode, setNodes]);
    return {scalingMode};
};