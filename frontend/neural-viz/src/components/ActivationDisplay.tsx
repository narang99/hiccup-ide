import { useEffect, useState } from 'react';
import { type ActivationData } from '../fetchers/activation';

interface ActivationDisplayProps {
  coordinate: string;
  fetcher: (coordinate: string) => Promise<ActivationData>;
  className?: string;
  maxSize?: number; // Maximum display size in pixels
}

export const ActivationDisplay = ({ 
  coordinate, 
  fetcher, 
  className = "", 
  maxSize = 60 
}: ActivationDisplayProps) => {
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadActivation = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const data = await fetcher(coordinate);
        if (!cancelled) {
          setActivationData(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load activation');
          setActivationData(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    loadActivation();

    return () => {
      cancelled = true;
    };
  }, [coordinate, fetcher]);

  const renderActivation = () => {
    if (isLoading) {
      return (
        <div 
          className="bg-gray-300 animate-pulse rounded"
          style={{ width: maxSize, height: maxSize }}
        />
      );
    }

    if (error || !activationData) {
      return (
        <div 
          className="bg-red-200 border border-red-400 rounded flex items-center justify-center text-xs text-red-700"
          style={{ width: maxSize, height: maxSize }}
        >
          ⚠️
        </div>
      );
    }

    // Handle 2D activation data (activation maps)
    if (Array.isArray(activationData.data) && activationData.shape.length === 2) {
      const [height, width] = activationData.shape;
      const data = activationData.data as number[][];
      
      // Find min/max for normalization
      const flatData = data.flat();
      const min = Math.min(...flatData);
      const max = Math.max(...flatData);
      const range = max - min;
      
      return (
        <div 
          className="border rounded overflow-hidden"
          style={{ width: maxSize, height: maxSize }}
        >
          <svg width={maxSize} height={maxSize} viewBox={`0 0 ${width} ${height}`}>
            {data.map((row, y) =>
              row.map((value, x) => {
                const intensity = range === 0 ? 0.5 : (value - min) / range;
                return (
                  <rect
                    key={`${x}-${y}`}
                    x={x}
                    y={y}
                    width={1}
                    height={1}
                    fill={`rgb(${Math.round(intensity * 255)}, ${Math.round(intensity * 255)}, ${Math.round(intensity * 255)})`}
                  />
                );
              })
            )}
          </svg>
        </div>
      );
    }

    // Handle scalar values (linear layer outputs)
    if (typeof activationData.data === 'number') {
      const value = activationData.data;
      const intensity = Math.max(0, Math.min(1, (value + 1) / 2)); // Assume range [-1, 1]
      
      return (
        <div 
          className="border rounded flex items-center justify-center text-xs font-bold"
          style={{ 
            width: maxSize, 
            height: maxSize,
            backgroundColor: `rgb(${Math.round(intensity * 255)}, ${Math.round(intensity * 255)}, ${Math.round(intensity * 255)})`
          }}
        >
          {value.toFixed(2)}
        </div>
      );
    }

    // Fallback for unknown data format
    return (
      <div 
        className="bg-gray-200 border border-gray-400 rounded flex items-center justify-center text-xs"
        style={{ width: maxSize, height: maxSize }}
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