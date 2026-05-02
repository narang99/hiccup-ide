import type { NodeFetchers, FetcherType } from "../../fetchers";
import { Handle, Position } from '@xyflow/react';
import { ActivationDisplay } from "../ActivationDisplay";
import { Link } from 'react-router-dom';
import { type HandleDirection } from "./ActivationFlowNode";

interface BaseActivationNodeProps {
    coordinate: string;
    fetchers?: NodeFetchers;
    fetcherType?: FetcherType;
    maxSize?: number;
    className?: string;
    title: string;
    badgeLabel?: string;
    badgeColor?: string;
    handleDirection?: HandleDirection;
    link?: string;
}

export default function BaseActivationNode({
    coordinate,
    fetchers,
    fetcherType = "activation",
    maxSize,
    className = "",
    title,
    badgeLabel,
    badgeColor = '#60a5fa',
    handleDirection = "TB",
    link
}: BaseActivationNodeProps) {
    const fetcher = fetchers?.[fetcherType];

    const handleStyle = {
        background: badgeColor,
        border: '2px solid #fff',
        width: '8px',
        height: '8px',
    };

    const content = (
        <div className={className} style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', position: 'relative' }}>
            {handleDirection === "TB" && (
                <>
                    <Handle type="target" position={Position.Top} style={handleStyle} />
                    <Handle type="source" position={Position.Bottom} style={handleStyle} />
                </>
            )}
            {handleDirection === "LR" && (
                <>
                    <Handle type="target" position={Position.Left} style={handleStyle} />
                    <Handle type="source" position={Position.Right} style={handleStyle} />
                </>
            )}
            {/* ── Toolbar ── */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 8px',
                background: 'rgba(0, 0, 0, 0.55)',
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '5px 5px 0 0',
            }}>
                {badgeLabel && (
                    <>
                        <span style={{
                            fontSize: 9,
                            fontWeight: 700,
                            letterSpacing: '0.06em',
                            textTransform: 'uppercase' as const,
                            color: badgeColor,
                            lineHeight: 1,
                        }}>
                            {badgeLabel}
                        </span>
                        <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 9, lineHeight: 1 }}>·</span>
                    </>
                )}
                <span style={{
                    fontSize: 9,
                    fontWeight: 500,
                    color: 'rgba(255,255,255,0.65)',
                    lineHeight: 1,
                }}>
                    {title}
                </span>
            </div>
            {/* ── Activation map (main area) ── */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 6,
                background: '#0d0d14',
                borderRadius: '0 0 5px 5px',
                cursor: 'pointer',
            }}>
                {fetcher ? (
                    <ActivationDisplay
                        coordinate={coordinate}
                        fetcher={fetcher}
                        maxSize={maxSize}
                    />
                ) : (
                    <div style={{
                        width: maxSize || "100%",
                        height: maxSize || "100%",
                        background: '#1e1e2e',
                        borderRadius: 4,
                        border: '1px solid rgba(255,255,255,0.08)',
                    }} />
                )}
            </div>
        </div>
    );

    if (link) {
        return (
            <Link to={link} style={{ textDecoration: 'none', color: 'inherit', display: 'block', width: '100%', height: '100%' }}>
                {content}
            </Link>
        );
    }

    return content;
}