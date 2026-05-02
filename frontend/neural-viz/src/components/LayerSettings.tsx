import { useLayerSettingsStore } from '../stores/layerSettingsStore';
import { useSelectedNodeStore } from '../stores/selectedNodeStore';
import { useSaliencyCacheStore } from '../stores/saliencyCacheStore';
import { getTopKThreshold } from '../utils/topk';
import { type LayerSaliencyData } from '../fetchers/saliency_map';
import { useCallback, useEffect, useMemo } from 'react';
import { useReactFlow } from '@xyflow/react';
import { DebouncedSlider } from './DebouncedSlider';

interface LayerSettingsViewProps {
  disabled: boolean;
  sliderValue: number;
  onSliderDebouncedChange: (value: number) => void;
  nodeId?: string;
  nodeData?: {
    label: string;
    layerType: string;
    nodeCount?: number;
  };
}

const LayerSettingsView = ({ disabled, sliderValue, onSliderDebouncedChange, nodeId, nodeData }: LayerSettingsViewProps) => {
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

export const LayerSettings = () => {
  const { selectedNode } = useSelectedNodeStore();
  const { getLayerSettings, updateSliderValue } = useLayerSettingsStore();
  const { fetchAndCacheLayerSaliency, clearCache } = useSaliencyCacheStore();
  const { getNodes } = useReactFlow();
  
  const nodeId = selectedNode?.id;
  const nodeData = selectedNode?.data;
  const disabled = !selectedNode;
  
  const sliderValue = nodeId ? getLayerSettings(nodeId).sliderValue : 100;

  // Get child nodes coordinates for the selected layer
  const childCoordinates = useMemo(() => {
    if (!nodeId) return [];
    
    const allNodes = getNodes();
    const childNodes = allNodes.filter(node => node.parentId === nodeId).filter(node => node.type === "ActivationFlowNode");
  return childNodes.map(node => node.data.coordinate as string);
  }, [nodeId, getNodes]);

  // Filter saliency data to only include child coordinates
  const filterSaliencyData = useCallback((data: LayerSaliencyData, coordinates: string[]): LayerSaliencyData => {
    const coordinateSet = new Set(coordinates);
    const filteredMaps = data.saliency_maps.filter(map => coordinateSet.has(map.coordinate));
    
    console.log(`Filtered saliency maps: ${data.saliency_maps.length} → ${filteredMaps.length}`);
    console.log('Child coordinates:', coordinates);
    console.log('Keeping coordinates:', filteredMaps.map(m => m.coordinate));
    
    return {
      ...data,
      saliency_maps: filteredMaps
    };
  }, []);

  const computeThreshold = useCallback((filteredData: LayerSaliencyData, thresholdRatio: number) => {
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
      return;
    }

    // Compute threshold using the utility function
    const threshold = getTopKThreshold(allValues, thresholdRatio / 100);
    console.log(`Computed threshold for ${filteredData.layer_name} at ${thresholdRatio}% (${allValues.length} values):`, threshold);
  }, []);

  // Debounced handler for expensive operations
  const handleSliderDebouncedChange = useCallback(async (value: number) => {
    if (!nodeId || !nodeData) return;
    
    // Update the store with the new value
    updateSliderValue(nodeId, value);
    
    try {
      // Extract layer name from node ID (e.g., "layers.2" from "layers.2.something")
      const layerName = nodeId.split('.').slice(0, 2).join('.');
      
      // Fetch saliency data using cache store
      const saliencyData = await fetchAndCacheLayerSaliency('example-model', 'first-input', layerName);
      
      // Filter data by child coordinates and compute threshold
      const filteredData = filterSaliencyData(saliencyData, childCoordinates);
      computeThreshold(filteredData, value);
    } catch (err) {
      console.error('Failed to fetch saliency data or compute threshold:', err);
    }
  }, [nodeId, nodeData, updateSliderValue, childCoordinates, filterSaliencyData, computeThreshold, fetchAndCacheLayerSaliency]);

  // Clear cache when selected node changes
  useEffect(() => {
    clearCache();
  }, [nodeId, clearCache]);

  return (
    <LayerSettingsView
      disabled={disabled}
      sliderValue={sliderValue}
      onSliderDebouncedChange={handleSliderDebouncedChange}
      nodeId={nodeId}
      nodeData={nodeData}
    />
  );
};