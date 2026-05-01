import { useLayerSettingsStore } from '../stores/layerSettingsStore';
import { useSelectedNode } from '../hooks/useSelectedNode';

export const LayerSettings = () => {
  const { selectedNode } = useSelectedNode();
  const { getLayerSettings, updateSliderValue } = useLayerSettingsStore();
  
  const nodeId = selectedNode?.id;
  const nodeData = selectedNode?.data;
  const disabled = !selectedNode;
  
  const sliderValue = nodeId ? getLayerSettings(nodeId).sliderValue : 50;

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    if (nodeId) {
      updateSliderValue(nodeId, value);
    }
  };

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
          <input
            type="range"
            min="0"
            max="100"
            value={sliderValue}
            onChange={handleSliderChange}
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