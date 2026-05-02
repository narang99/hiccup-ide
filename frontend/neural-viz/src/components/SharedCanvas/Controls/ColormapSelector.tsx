import { COLORMAPS, COLORMAP_META, type ColormapName } from '../../../utils/colormaps';
import { useColormap } from '../../../hooks/useColormap';

const COLORMAP_KEYS = Object.keys(COLORMAPS) as ColormapName[];

export const ColormapSelector = () => {
  const { colormap: activeColormap, setColormap, scalingMode, setScalingMode } = useColormap();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      padding: '7px 10px',
      background: 'rgba(13, 13, 20, 0.88)',
      border: '1px solid rgba(255,255,255,0.09)',
      borderRadius: 10,
      backdropFilter: 'blur(10px)',
      boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
    }}>
      {/* First row: Colormap selection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.35)',
          marginRight: 2,
          minWidth: '60px',
        }}>
          Colormap
        </span>

        {COLORMAP_KEYS.map((key) => {
          const isActive = key === activeColormap;
          const meta = COLORMAP_META[key];
          return (
            <button
              key={key}
              onClick={() => setColormap(key)}
              title={key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 9px',
                borderRadius: 6,
                border: isActive
                  ? '1px solid rgba(255,255,255,0.3)'
                  : '1px solid rgba(255,255,255,0.07)',
                background: isActive
                  ? 'rgba(255,255,255,0.12)'
                  : 'transparent',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {/* Gradient swatch */}
              <span style={{
                display: 'inline-block',
                width: 28,
                height: 8,
                borderRadius: 3,
                background: meta.gradient,
                flexShrink: 0,
              }} />
              {/* Label */}
              <span style={{
                fontSize: 10,
                fontWeight: isActive ? 700 : 400,
                color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
                lineHeight: 1,
              }}>
                {meta.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Second row: Scaling selection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.35)',
          marginRight: 2,
          minWidth: '60px',
        }}>
          Scaling
        </span>

        {(['local', 'global'] as const).map((mode) => {
          const isActive = mode === scalingMode;
          return (
            <button
              key={mode}
              onClick={() => setScalingMode(mode)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 9px',
                borderRadius: 6,
                border: isActive
                  ? '1px solid rgba(255,255,255,0.3)'
                  : '1px solid rgba(255,255,255,0.07)',
                background: isActive
                  ? 'rgba(255,255,255,0.12)'
                  : 'transparent',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                flex: 1,
                justifyContent: 'center',
              }}
            >
              <span style={{
                fontSize: 10,
                fontWeight: isActive ? 700 : 400,
                color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
                lineHeight: 1,
                textTransform: 'capitalize',
              }}>
                {mode}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
