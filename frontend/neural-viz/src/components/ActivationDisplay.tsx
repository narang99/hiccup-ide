import { useRef, useCallback } from 'react';
import chroma from 'chroma-js';
import { type ActivationData } from '../fetchers/activation';
import { type LayerSaliencyMap } from '../fetchers/saliency_map';
import { COLORMAPS, COLORMAP_META, normalizeSymmetric, type ColormapName } from '../utils/colormaps';
import { useColormap } from '../hooks/useColormap';
import { filterActivation } from '../activationFiltering';
import type { ActivationFilterAlgorithm } from '../types/activationFiltering';
import type { OverlayAlgorithm } from '../types/overlay';

type BaseData = ActivationData | LayerSaliencyMap;

interface ActivationDisplayProps {
  activationData: BaseData | null;
  isLoading: boolean;
  error: string | null;
  className?: string;
  maxSize?: number;
  /** Explicit override. If omitted, the global colormap from context is used. */
  colormap?: ColormapName;
  filterAlgorithm?: ActivationFilterAlgorithm;
  // color map, maximum value for opacity scaling
  absMax?: number;
  onPixelHover?: (gridCoord: [number, number], position: [number, number]) => void;
  onPixelLeave?: () => void;
  onPixelClick?: (gridCoord: [number, number] | null, position: [number, number] | null) => void;
  overlayAlgorithm?: OverlayAlgorithm;
}

export const ActivationDisplay = ({
  activationData,
  isLoading,
  error,
  className = '',
  colormap,
  filterAlgorithm = { type: 'Id' },
  absMax,
  onPixelHover,
  onPixelLeave,
  onPixelClick,
  overlayAlgorithm = { type: 'NoOverlay' },
}: ActivationDisplayProps) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Fall back to global context colormap when no explicit prop is passed
  const { colormap: globalColormap } = useColormap();
  const resolvedColormap: ColormapName = colormap ?? globalColormap;
  const defaultRectColor = resolvedColormap === "rd_bk_gn" ? "#f59e0b" : "#000000";
  const scale = COLORMAPS[resolvedColormap];

  const getCoordinates = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || !activationData) return null;

    const [height, width] = activationData.shape;
    if (height === 0 || width === 0) return null;

    const svg = svgRef.current;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;

    const cursorPt = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    
    // Map SVG coordinates to grid coordinates
    const gridX = Math.floor(cursorPt.x);
    const gridY = Math.floor(cursorPt.y);

    const isWithinBounds = gridX >= 0 && gridX < width && gridY >= 0 && gridY < height;

    return {
      gridX,
      gridY,
      isWithinBounds,
      position: [cursorPt.x, cursorPt.y] as [number, number]
    };
  }, [activationData]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!onPixelHover) return;
    const coords = getCoordinates(e);
    if (!coords) return;

    if (coords.isWithinBounds) {
      onPixelHover([coords.gridX, coords.gridY], coords.position);
    } else if (onPixelLeave) {
      onPixelLeave();
    }
  }, [getCoordinates, onPixelHover, onPixelLeave]);

  const handleMouseClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!onPixelClick) return;
    e.stopPropagation();
    const coords = getCoordinates(e);
    
    if (!coords) {
        // If we can't get coordinates (e.g. data not loaded), we still might want to trigger the click with nulls
        onPixelClick(null, null);
        return;
    }

    if (coords.isWithinBounds) {
        onPixelClick([coords.gridX, coords.gridY], coords.position);
    } else {
        onPixelClick(null, coords.position);
    }
  }, [getCoordinates, onPixelClick]);

  const renderActivation = (absMax?: number) => {
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
          onClick={(e) => {
            e.stopPropagation();
            onPixelClick?.(null, null);
          }}
        >
          ⚠️
        </div>
      );
    }

    // ── 2-D activation map ──────────────────────────────────────────────────
    if (Array.isArray(activationData.data) && activationData.shape.length === 2) {
      const [height, width] = activationData.shape;
      const rawData = activationData.data as number[][];
      const data = filterActivation(rawData, filterAlgorithm);

      const flatData = data.flat();
      const computeAbsMax = (data: number[]) => (Math.max(Math.abs(Math.min(...data)),Math.abs(Math.max(...data))) );

      const actualAbsMax = (
        (absMax !== undefined) 
        ? absMax : 
        (flatData.length > 0 ? computeAbsMax(flatData) : 1)
      );

      return (
        <div style={{ 
          width: "100%", 
          height: "100%", 
          borderRadius: 4, 
          overflow: 'hidden',
          position: 'relative',
          boxSizing: 'border-box'
        }}>
          <svg 
            ref={svgRef}
            width={"100%"} 
            height={"100%"} 
            viewBox={`0 0 ${width} ${height}`} 
            preserveAspectRatio="none"
            onMouseMove={handleMouseMove}
            onMouseLeave={onPixelLeave}
            onClick={handleMouseClick}
          >
            {data.map((row, y) =>
              row.map((value, x) => (
                <rect
                  key={`${x}-${y}`}
                  x={x}
                  y={y}
                  width={1}
                  height={1}
                  fill={scale(normalizeSymmetric(value, actualAbsMax)).hex()}
                />
              ))
            )}
            {overlayAlgorithm.type === 'DrawRect' && (
              <rect
                x={overlayAlgorithm.start[0]}
                y={overlayAlgorithm.start[1]}
                width={overlayAlgorithm.end[0] - overlayAlgorithm.start[0]}
                height={overlayAlgorithm.end[1] - overlayAlgorithm.start[1]}
                fill="none"
                stroke={overlayAlgorithm.color || defaultRectColor}
                strokeWidth={0.5}
                style={{ vectorEffect: 'non-scaling-stroke' }}
              />
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
            position: 'relative',
            boxSizing: 'border-box'
          }}
          onClick={(e) => {
            e.stopPropagation();
            onPixelClick?.(null, null);
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
          position: 'relative',
          boxSizing: 'border-box'
        }}
        onClick={(e) => {
          e.stopPropagation();
          onPixelClick?.(null, null);
        }}
      >
        ?
      </div>
    );
  };

  return (
    <div className={`inline-block ${className}`}>
      {renderActivation(absMax)}
    </div>
  );
};

// Re-export for convenience
export type { ColormapName };
export { COLORMAPS, COLORMAP_META };