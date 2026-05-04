import { useMemo, useState, useEffect, useCallback, useRef } from 'react';
import { useNodes, useReactFlow } from '@xyflow/react';
import { useSaliencyCacheStore } from '../../stores/saliencyCacheStore';
import { useFetcherType } from '../../hooks/useFetcherType';
import { useAliases } from '../../hooks/useAliases';
import type { SelectedNode } from '../../types/node';
import type { ActivationFilterAlgorithm } from '../../types/activationFiltering';
import { HistogramPlot, type HistogramData, type HistogramStats } from './HistogramPlot';

interface PruneHistogramPreviewProps {
    selectedNode: SelectedNode | null;
    onChangeThreshold?: (layerId: string, threshold: number, algorithm: ActivationFilterAlgorithm) => void;
}

export const PruneHistogramPreview = ({ selectedNode, onChangeThreshold }: PruneHistogramPreviewProps) => {
    const { fetcherType } = useFetcherType();
    const { fetchAndCacheBatchSaliency } = useSaliencyCacheStore();
    const { modelAlias, inputAlias } = useAliases();
    const { setNodes } = useReactFlow();
    const nodes = useNodes();

    const [threshold, setThreshold] = useState<number>(0);
    const [allValues, setAllValues] = useState<number[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    const nodeId = selectedNode?.id;
    const nodeData = selectedNode?.data;

    const childNodes = useMemo(() => {
        if (!nodeId) return [];
        return nodes.filter(node => node.parentId === nodeId && node.type === "ActivationFlowNode");
    }, [nodeId, nodes]);

    const childCoordinates = useMemo(() => {
        return childNodes.map(node => node.data.coordinate as string);
    }, [childNodes]);

    const coordsKey = useMemo(() => JSON.stringify(childCoordinates), [childCoordinates]);
    const lastNodeIdRef = useRef<string | null>(null);

    // Fetch data when selection changes
    useEffect(() => {
        const fetchData = async () => {
            if (!nodeId || childCoordinates.length === 0 || fetcherType !== 'saliency_map') {
                setAllValues([]);
                return;
            }

            setIsLoading(true);
            try {
                const saliencyData = await fetchAndCacheBatchSaliency(
                    modelAlias,
                    inputAlias,
                    childCoordinates,
                    nodeId
                );

                const values: number[] = [];
                saliencyData.items.forEach(map => {
                    map.data.forEach(row => {
                        if (Array.isArray(row)) {
                            values.push(...row.filter(v => v > 0));
                        }
                    });
                });
                
                values.sort((a, b) => a - b);
                setAllValues(values);
                
                // Only reset threshold if the layer actually changed
                if (lastNodeIdRef.current !== nodeId) {
                    setThreshold(0);
                    lastNodeIdRef.current = nodeId;
                }
            } catch (err) {
                console.error('Failed to fetch saliency data:', err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
        // Use coordsKey to only trigger when the list of coordinates actually changes
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nodeId, coordsKey, fetcherType, fetchAndCacheBatchSaliency]);

    // Data Aggregation for Histogram
    const histogramData = useMemo<HistogramData | null>(() => {
        if (allValues.length === 0) return null;

        const min = allValues[0];
        const max = allValues[allValues.length - 1];
        const totalSum = allValues.reduce((a, b) => a + b, 0);
        
        const binCount = 40;
        const binWidth = (max - min) / binCount;
        const bins = new Array(binCount).fill(0);
        
        allValues.forEach(v => {
            const binIdx = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
            bins[binIdx]++;
        });

        const cdf: { x: number; energy: number }[] = [];
        for (let i = 0; i <= 100; i++) {
            const t = min + (max - min) * (i / 100);
            let sumBelow = 0;
            for(let j = 0; j < allValues.length; j++) {
                if (allValues[j] < t) sumBelow += allValues[j];
                else break;
            }
            cdf.push({ x: t, energy: (totalSum - sumBelow) / totalSum });
        }

        return { min, max, bins, maxBin: Math.max(...bins), totalSum, cdf };
    }, [allValues]);

    // Statistics for the current threshold
    const stats = useMemo<HistogramStats | null>(() => {
        if (!histogramData || allValues.length === 0) return null;

        let sumBelow = 0;
        let countBelow = 0;
        for (let i = 0; i < allValues.length; i++) {
            if (allValues[i] < threshold) {
                sumBelow += allValues[i];
                countBelow++;
            } else {
                break;
            }
        }

        return {
            energyRetained: ((histogramData.totalSum - sumBelow) / histogramData.totalSum * 100).toFixed(1),
            prunedRatio: ((countBelow / allValues.length) * 100).toFixed(1)
        };
    }, [histogramData, allValues, threshold]);

    const handleThresholdChange = useCallback((newThreshold: number) => {
        if (!nodeId) return;
        setThreshold(newThreshold);

        const algorithm: ActivationFilterAlgorithm = {
            type: "ThresholdAlgorithm",
            threshold: newThreshold
        };

        setNodes((nodes) => nodes.map((node) => {
            if (node.parentId === nodeId && node.type === "ActivationFlowNode") {
                return {
                    ...node,
                    data: {
                        ...node.data,
                        filterAlgorithm: algorithm
                    }
                };
            }
            return node;
        }));

        if (onChangeThreshold) {
            onChangeThreshold(nodeId, newThreshold, algorithm);
        }
    }, [nodeId, setNodes, onChangeThreshold]);

    if (!selectedNode || fetcherType !== 'saliency_map') return null;

    return (
        <HistogramPlot
            data={histogramData}
            threshold={threshold}
            onThresholdChange={handleThresholdChange}
            stats={stats}
            isLoading={isLoading}
            subtitle={nodeData?.label}
        />
    );
};
