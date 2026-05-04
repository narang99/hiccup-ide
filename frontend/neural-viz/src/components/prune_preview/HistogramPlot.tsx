import React, { useRef, useState, useEffect } from 'react';

export interface HistogramData {
    min: number;
    max: number;
    bins: number[];
    maxBin: number;
    totalSum: number;
    cdf: { x: number; energy: number }[];
}

export interface HistogramStats {
    energyRetained: string;
    prunedRatio: string;
}

interface HistogramPlotProps {
    data: HistogramData | null;
    threshold: number;
    onThresholdChange: (threshold: number) => void;
    stats?: HistogramStats | null;
    isLoading?: boolean;
    title?: string;
    subtitle?: string;
}

export const HistogramPlot = ({
    data,
    threshold,
    onThresholdChange,
    stats,
    isLoading,
    title,
    subtitle
}: HistogramPlotProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleInteraction = (clientX: number) => {
        if (!containerRef.current || !data) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const percentage = x / rect.width;
        const newThreshold = data.min + (data.max - data.min) * percentage;
        onThresholdChange(newThreshold);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging) handleInteraction(e.clientX);
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (isDragging) handleInteraction(e.touches[0].clientX);
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true);
        handleInteraction(e.clientX);
    };

    const handleTouchStart = (e: React.TouchEvent) => {
        setIsDragging(true);
        handleInteraction(e.touches[0].clientX);
    };

    const stopDragging = () => setIsDragging(false);

    useEffect(() => {
        if (isDragging) {
            window.addEventListener('mouseup', stopDragging);
            window.addEventListener('touchend', stopDragging);
        }
        return () => {
            window.removeEventListener('mouseup', stopDragging);
            window.removeEventListener('touchend', stopDragging);
        };
    }, [isDragging]);

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
            width: '280px',
            color: '#fff',
            userSelect: 'none'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase' }}>
                    {title || 'Saliency Distribution'}
                </span>
                {subtitle && (
                    <span style={{ fontSize: 10, background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: 4 }}>
                        {subtitle}
                    </span>
                )}
            </div>

            {isLoading ? (
                <div style={{ height: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>
                    Computing histogram...
                </div>
            ) : data ? (
                <>
                    <div 
                        ref={containerRef}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onTouchStart={handleTouchStart}
                        onTouchMove={handleTouchMove}
                        style={{ 
                            height: '120px', 
                            position: 'relative', 
                            cursor: 'crosshair',
                            background: 'rgba(0,0,0,0.2)',
                            borderRadius: 4,
                            overflow: 'hidden'
                        }}
                    >
                        <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
                            {/* Histogram Bars */}
                            {data.bins.map((count, i) => {
                                const height = (count / data.maxBin) * 100;
                                return (
                                    <rect
                                        key={i}
                                        x={(i / data.bins.length) * 100}
                                        y={100 - height}
                                        width={100 / data.bins.length}
                                        height={height}
                                        fill="rgba(168, 85, 247, 0.3)"
                                        stroke="rgba(168, 85, 247, 0.1)"
                                        strokeWidth="0.1"
                                    />
                                );
                            })}

                            {/* CDF Energy Curve */}
                            <path
                                d={`M ${data.cdf.map((p, i) => {
                                    const x = (i / (data.cdf.length - 1)) * 100;
                                    const y = 100 - (p.energy * 100);
                                    return `${x} ${y}`;
                                }).join(' L ')}`}
                                fill="none"
                                stroke="#22c55e"
                                strokeWidth="1"
                                opacity="0.6"
                                vectorEffect="non-scaling-stroke"
                            />

                            {/* Threshold Line */}
                            <line
                                x1={((threshold - data.min) / (data.max - data.min)) * 100}
                                y1="0"
                                x2={((threshold - data.min) / (data.max - data.min)) * 100}
                                y2="100"
                                stroke="#fff"
                                strokeWidth="1"
                                strokeDasharray="2 1"
                                vectorEffect="non-scaling-stroke"
                            />
                        </svg>

                        {/* Labels for Histogram */}
                        <div style={{ position: 'absolute', bottom: 2, left: 4, fontSize: 8, color: 'rgba(255,255,255,0.3)' }}>{data.min.toFixed(4)}</div>
                        <div style={{ position: 'absolute', bottom: 2, right: 4, fontSize: 8, color: 'rgba(255,255,255,0.3)' }}>{data.max.toFixed(4)}</div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', marginBottom: 2 }}>Energy Retained</div>
                            <div style={{ fontSize: 14, fontWeight: 700, color: '#22c55e' }}>{stats?.energyRetained || '0.0'}%</div>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', marginBottom: 2 }}>Pruned Ratio</div>
                            <div style={{ fontSize: 14, fontWeight: 700, color: '#ef4444' }}>{stats?.prunedRatio || '0.0'}%</div>
                        </div>
                    </div>

                    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)', textAlign: 'center' }}>
                        Threshold: <span style={{ color: '#fff', fontFamily: 'monospace' }}>{threshold.toExponential(3)}</span>
                    </div>
                </>
            ) : (
                <div style={{ height: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>
                    No data available
                </div>
            )}
        </div>
    );
};
