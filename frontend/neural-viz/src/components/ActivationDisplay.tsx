import { useEffect, useState } from 'react';
import chroma from 'chroma-js';
import { type ActivationData } from '../fetchers/activation';
import { COLORMAPS, COLORMAP_META, normalizeSymmetric, type ColormapName } from '../utils/colormaps';
import { useColormap } from '../hooks/useColormap';

interface ActivationDisplayProps {
  coordinate: string;
  fetcher: (coordinate: string) => Promise<ActivationData>;
  className?: string;
  maxSize?: number;
  /** Explicit override. If omitted, the global colormap from context is used. */
  colormap?: ColormapName;
}

export const ActivationDisplay = ({
  coordinate,
  fetcher,
  className = '',
  colormap,
}: ActivationDisplayProps) => {
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fall back to global context colormap when no explicit prop is passed
  const { colormap: globalColormap } = useColormap();
  const resolvedColormap: ColormapName = colormap ?? globalColormap;
  const scale = COLORMAPS[resolvedColormap];

  useEffect(() => {
    let cancelled = false;

    const loadActivation = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const data = await fetcher(coordinate);
        if (!cancelled) setActivationData(data);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load activation');
          setActivationData(null);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    loadActivation();
    return () => { cancelled = true; };
  }, [coordinate, fetcher]);

  const renderActivation = () => {
    if (isLoading) {
      return (
        <div
          style={{
            width: "100%",
            height: "100%",
            background: 'rgba(255,255,255,0.06)',
            borderRadius: 4,
          }}
        />
      );
    }

    if (error || !activationData) {
      return (
        <div
          style={{
            width: "100%",
            height: "100%",
            background: 'rgba(255,80,80,0.15)',
            border: '1px solid rgba(255,80,80,0.4)',
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
          }}
        >
          ⚠️
        </div>
      );
    }

    // ── 2-D activation map ──────────────────────────────────────────────────
    if (Array.isArray(activationData.data) && activationData.shape.length === 2) {
      const [height, width] = activationData.shape;
      const data = activationData.data as number[][];

      const flatData = data.flat();
      const absMax = Math.max(
        Math.abs(Math.min(...flatData)),
        Math.abs(Math.max(...flatData)),
      );

      return (
        <div style={{ width: "100%", height: "100%", borderRadius: 4, overflow: 'hidden' }}>
          <svg width={"100%"} height={"100%"} viewBox={`0 0 ${width} ${height}`}>
            {data.map((row, y) =>
              row.map((value, x) => (
                <rect
                  key={`${x}-${y}`}
                  x={x}
                  y={y}
                  width={1}
                  height={1}
                  fill={scale(normalizeSymmetric(value, absMax)).hex()}
                />
              ))
            )}
          </svg>
        </div>
      );
    }

    // ── Scalar value (e.g. linear layer output) ─────────────────────────────
    if (typeof activationData.data === 'number') {
      const value = activationData.data;
      const t = normalizeSymmetric(value, 3); // ±3 as default range
      const bg = scale(t).hex();
      const textColor = chroma(bg).luminance() > 0.35 ? '#000' : '#fff';

      return (
        <div
          style={{
            width: "100%",
            height: "100%",
            background: bg,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            fontWeight: 700,
            color: textColor,
          }}
        >
          {value.toFixed(2)}
        </div>
      );
    }

    // ── Fallback ────────────────────────────────────────────────────────────
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          color: 'rgba(255,255,255,0.4)',
        }}
      >
        ?
      </div>
    );
  };

  return (
    <div className={`inline-block ${className}`}>
      {renderActivation()}
    </div>
  );
};

// Re-export for convenience
export type { ColormapName };
export { COLORMAPS, COLORMAP_META };