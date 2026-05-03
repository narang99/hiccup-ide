import { useLayerSettingsStore } from '../../stores/layerSettingsStore';
import { useSaliencyCacheStore } from '../../stores/saliencyCacheStore';
import { getTopKThreshold } from '../../utils/topk';
import { type LayerSaliencyData } from '../../fetchers/saliency_map';
import { loadWorkflowThresholds, saveWorkflowThreshold } from '../../fetchers/threshold';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { DebouncedSlider } from '../DebouncedSlider';
import { useFetcherType } from '../../hooks/useFetcherType';
import { type ActivationFilterAlgorithm } from '../../activationFiltering/types';
import type { SelectedNode } from '../../types/node';

interface LayerSettingsViewProps {
    disabled: boolean;
    sliderValue: number;
    onSliderDebouncedChange: (value: number) => void;
    onSaveThreshold: () => void;
    isSaving: boolean;
    nodeId?: string;
    nodeData?: {
        label: string;
        layerType: string;
        nodeCount?: number;
    };
}

const LayerSettingsView = ({ disabled, sliderValue, onSliderDebouncedChange, onSaveThreshold, isSaving, nodeId, nodeData }: LayerSettingsViewProps) => {
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            padding: '16px',
            background: 'rgba(13, 13, 20, 0.88)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 10,
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            minWidth: '200px',
            opacity: disabled ? 0.5 : 1,
            pointerEvents: disabled ? 'none' : 'auto',
            transition: 'opacity 0.2s ease',
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '4px',
            }}>
                <span style={{
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    color: 'rgba(255,255,255,0.35)',
                }}>
                    Layer Settings
                </span>
                {nodeData && (
                    <span style={{
                        fontSize: 10,
                        fontWeight: 500,
                        color: 'rgba(255,255,255,0.6)',
                        background: 'rgba(255,255,255,0.1)',
                        padding: '2px 6px',
                        borderRadius: 4,
                    }}>
                        {nodeData.layerType}
                    </span>
                )}
            </div>

            {/* Slider Section */}
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}>
                    <span style={{
                        fontSize: 11,
                        fontWeight: 500,
                        color: 'rgba(255,255,255,0.7)',
                    }}>
                        Intensity
                    </span>
                    <span style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: '#fff',
                        background: 'rgba(255,255,255,0.1)',
                        padding: '2px 6px',
                        borderRadius: 4,
                        minWidth: '30px',
                        textAlign: 'center',
                    }}>
                        {sliderValue}
                    </span>
                </div>

                <div style={{ position: 'relative' }}>
                    <DebouncedSlider
                        key={nodeId}
                        initialValue={sliderValue}
                        min={0}
                        max={100}
                        disabled={disabled}
                        onDebouncedChange={onSliderDebouncedChange}
                        style={{
                            width: '100%',
                            height: '4px',
                            background: `linear-gradient(to right, 
                rgba(96, 165, 250, 0.8) 0%, 
                rgba(96, 165, 250, 0.8) ${sliderValue}%, 
                rgba(255,255,255,0.2) ${sliderValue}%, 
                rgba(255,255,255,0.2) 100%)`,
                            outline: 'none',
                            borderRadius: '2px',
                            appearance: 'none',
                            cursor: disabled ? 'not-allowed' : 'pointer',
                        }}
                    />
                </div>

                {/* Value indicators */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginTop: '4px',
                }}>
                    <span style={{
                        fontSize: 9,
                        color: 'rgba(255,255,255,0.4)',
                        fontWeight: 500,
                    }}>
                        0
                    </span>
                    <span style={{
                        fontSize: 9,
                        color: 'rgba(255,255,255,0.4)',
                        fontWeight: 500,
                    }}>
                        100
                    </span>
                </div>
            </div>

            {/* Save Button */}
            <div style={{
                marginTop: '12px',
            }}>
                <button
                    onClick={onSaveThreshold}
                    disabled={disabled || isSaving}
                    style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: disabled || isSaving ? 'rgba(255,255,255,0.1)' : 'rgba(34, 197, 94, 0.2)',
                        border: disabled || isSaving ? '1px solid rgba(255,255,255,0.2)' : '1px solid rgba(34, 197, 94, 0.5)',
                        borderRadius: 6,
                        color: disabled || isSaving ? 'rgba(255,255,255,0.4)' : '#22c55e',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: disabled || isSaving ? 'not-allowed' : 'pointer',
                        transition: 'all 0.2s ease',
                    }}
                >
                    {isSaving ? 'Saving...' : 'Save Threshold'}
                </button>
            </div>

            {/* Layer Info */}
            {nodeData && (
                <div style={{
                    marginTop: '8px',
                    padding: '8px',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: 6,
                    border: '1px solid rgba(255,255,255,0.1)',
                }}>
                    <div style={{
                        fontSize: 10,
                        color: 'rgba(255,255,255,0.6)',
                        marginBottom: '4px',
                    }}>
                        Layer: {nodeData.label}
                    </div>
                    {nodeData.nodeCount && (
                        <div style={{
                            fontSize: 9,
                            color: 'rgba(255,255,255,0.5)',
                        }}>
                            {nodeData.nodeCount} {nodeData.layerType === 'Conv2d' ? 'kernels' : 'channels'}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};


export interface LayerSettingsProps {
    selectedNode: SelectedNode | null;
}

export const LayerSettings = ({ selectedNode }: LayerSettingsProps) => {
    const { fetcherType } = useFetcherType();
    const { getLayerSettings, updateSliderValue, loadSliderValuesFromThresholds } = useLayerSettingsStore();
    const { fetchAndCacheBatchSaliency, clearCache } = useSaliencyCacheStore();
    const { getNodes, setNodes } = useReactFlow();

    const [isSaving, setIsSaving] = useState(false);

    // Hardcoded values as mentioned in requirements
    const modelAlias = 'example-model';
    const inputAlias = 'first-input';
    const workflowName = 'default-workflow';

    const nodeId = selectedNode?.id;
    const nodeData = selectedNode?.data;

    // Disable slider if no node is selected OR if fetcher type is not saliency_map
    const disabled = !selectedNode || fetcherType !== 'saliency_map';

    const sliderValue = nodeId ? getLayerSettings(nodeId).sliderValue : 100;

    // Get child nodes for the selected layer
    const childNodes = useMemo(() => {
        if (!nodeId) return [];

        const allNodes = getNodes();
        return allNodes.filter(node => node.parentId === nodeId).filter(node => node.type === "ActivationFlowNode");
    }, [nodeId, getNodes]);


    const childCoordinates = useMemo(() => {
        return childNodes.map(node => node.data.coordinate as string);
    }, [childNodes]);

    const computeThreshold = useCallback((filteredData: LayerSaliencyData, thresholdRatio: number): number | undefined => {
        // Flatten all filtered saliency data into a single array
        const allValues: number[] = [];
        filteredData.saliency_maps.forEach(map => {
            map.data.forEach(row => {
                if (Array.isArray(row)) {
                    allValues.push(...row);
                }
            });
        });

        if (allValues.length === 0) {
            console.log('No saliency data to compute threshold');
            return undefined;
        }

        // Compute threshold using the utility function
        const threshold = getTopKThreshold(allValues, thresholdRatio / 100);
        console.log(`Computed threshold for ${filteredData.layer_name} at ${thresholdRatio}% (${allValues.length} values):`, threshold);
        return threshold;
    }, []);

    // Debounced handler for expensive operations
    const handleSliderDebouncedChange = useCallback(async (value: number) => {
        if (!nodeId || !nodeData) return;

        // Update the store with the new value
        updateSliderValue(nodeId, value);

        try {
            // Fetch saliency data for specific child coordinates using cache store
            // We use nodeId as the cache key
            const saliencyData = await fetchAndCacheBatchSaliency(
                'example-model',
                'first-input',
                childCoordinates,
                nodeId
            );

            // Compute threshold
            const threshold = computeThreshold(saliencyData, value);

            if (threshold !== undefined) {
                setNodes((nodes) => nodes.map((node) => {
                    if (node.parentId === nodeId && node.type === "ActivationFlowNode") {
                        return {
                            ...node,
                            data: {
                                ...node.data,
                                filterAlgorithm: {
                                    type: "ThresholdAlgorithm",
                                    threshold: threshold
                                }
                            }
                        };
                    }
                    return node;
                }));
            }
        } catch (err) {
            console.error('Failed to fetch saliency data or compute threshold:', err);
        }
    }, [nodeId, nodeData, updateSliderValue, childCoordinates, computeThreshold, fetchAndCacheBatchSaliency, setNodes]);

    // Load thresholds on component mount
    useEffect(() => {
        const loadThresholds = async () => {
            try {
                const thresholds = await loadWorkflowThresholds(modelAlias, inputAlias, workflowName);

                // Update slider values in store
                loadSliderValuesFromThresholds(thresholds);

                // Apply saved algorithms to nodes
                setNodes((nodes) => nodes.map((node) => {
                    if (node.type === "ActivationFlowNode" && node.parentId) {
                        // Find the threshold for this node's parent layer
                        const threshold = thresholds.find(t => t.layer_id === node.parentId);
                        if (threshold && threshold.algorithm && typeof threshold.algorithm === 'object' && 'type' in threshold.algorithm) {
                            console.log(`Applying saved algorithm to node ${node.id} (parent: ${node.parentId}):`, threshold.algorithm);
                            return {
                                ...node,
                                data: {
                                    ...node.data,
                                    filterAlgorithm: threshold.algorithm
                                }
                            };
                        }
                    }
                    return node;
                }));

                console.log(`Loaded ${thresholds.length} thresholds and applied algorithms to nodes`);
            } catch (error) {
                console.error('Failed to load thresholds:', error);
            }
        };

        loadThresholds();
    }, [loadSliderValuesFromThresholds, getNodes, setNodes]);

    // Clear cache when selected node changes
    useEffect(() => {
        clearCache();
    }, [nodeId, clearCache]);

    const handleSaveThreshold = useCallback(async () => {
        if (!nodeId) return;

        setIsSaving(true);
        try {
            console.log(`Saving threshold for layer ${nodeId}, current sliderValue: ${sliderValue}`);

            // Get the current algorithm from the nodes to save it
            const allNodes = getNodes();
            const currentNode = allNodes.find(node =>
                node.parentId === nodeId && node.type === "ActivationFlowNode"
            );

            const filterAlgorithm = currentNode?.data?.filterAlgorithm;
            const algorithm: ActivationFilterAlgorithm = (
                filterAlgorithm && typeof filterAlgorithm === 'object' && 'type' in filterAlgorithm
            ) ? filterAlgorithm as ActivationFilterAlgorithm : { type: "Id" };

            await saveWorkflowThreshold(modelAlias, inputAlias, workflowName, {
                layer_id: nodeId,
                slider_value: sliderValue,
                algorithm: algorithm
            });
            console.log(`Algorithm saved for layer ${nodeId}, slider_value: ${sliderValue}, algorithm:`, algorithm);
        } catch (error) {
            console.error('Failed to save algorithm:', error);
        } finally {
            setIsSaving(false);
        }
    }, [nodeId, sliderValue, getNodes]);

    return (
        <LayerSettingsView
            disabled={disabled}
            sliderValue={sliderValue}
            onSliderDebouncedChange={handleSliderDebouncedChange}
            onSaveThreshold={handleSaveThreshold}
            isSaving={isSaving}
            nodeId={nodeId}
            nodeData={nodeData}
        />
    );
};