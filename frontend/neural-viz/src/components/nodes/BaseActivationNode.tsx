import type { NodeFetchers, FetcherType } from "../../fetchers";
import { ActivationDisplay } from "../ActivationDisplay";
import { Link } from 'react-router-dom';
import { type HandleDirection, type ClickAction } from "./ActivationFlowNode";
import SingleOrNoHandle from "../SingleOrNoHandle";
import type { ActivationFilterAlgorithm } from "../../types/activationFiltering";
import type { OverlayAlgorithm } from "../../types/overlay";

import { useFetcherType } from "../../hooks/useFetcherType";
import { useCallback } from "react";
import { useNodeId } from "@xyflow/react";

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
    clickAction?: ClickAction;
    filterAlgorithm?: ActivationFilterAlgorithm;
    absMax?: number;
    onPixelHover?: (nodeId: string, coordinate: string, gridCoord: [number, number], position: [number, number]) => void;
    onPixelLeave?: (nodeId: string, coordinate: string) => void;
    overlayAlgorithm?: OverlayAlgorithm;
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
    clickAction,
    filterAlgorithm,
    absMax,
    onPixelHover,
    onPixelLeave,
    overlayAlgorithm
}: BaseActivationNodeProps) {
    const { workAlias, graphAlias } = useFetcherType();
    const nodeId = useNodeId();

    const fetcher = useCallback((coord: string) => {
        const f = fetchers?.[fetcherType];
        if (!f) return Promise.reject("No fetcher");
        
        return f(coord, workAlias || undefined, graphAlias || undefined);
    }, [fetchers, fetcherType, workAlias, graphAlias]);

    const handleHover = useCallback((gridCoord: [number, number], position: [number, number]) => {
        if (onPixelHover && nodeId) {
            onPixelHover(nodeId, coordinate, gridCoord, position);
        }
    }, [onPixelHover, nodeId, coordinate]);

    const handleLeave = useCallback(() => {
        if (onPixelLeave && nodeId) {
            onPixelLeave(nodeId, coordinate);
        }
    }, [onPixelLeave, nodeId, coordinate]);

    const handlePixelClick = useCallback((gridCoord: [number, number] | null, position: [number, number] | null) => {
        if (clickAction?.type === 'callback' && nodeId) {
            clickAction.callback(nodeId, coordinate, gridCoord, position);
        }
    }, [clickAction, nodeId, coordinate]);

    const handleGeneralClick = useCallback((e: React.MouseEvent) => {
        if (clickAction?.type === 'callback' && nodeId) {
            // Check if this was a click on the ActivationDisplay by checking if it was handled
            // If we use stopPropagation in ActivationDisplay, we don't need to check anything here.
            clickAction.callback(nodeId, coordinate, null, null);
        }
    }, [clickAction, nodeId, coordinate]);

    const content = (
        <div 
            className={className} 
            style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', position: 'relative' }}
            onClick={clickAction?.type === 'callback' ? handleGeneralClick : undefined}
        >
            <SingleOrNoHandle handleDirection={handleDirection} badgeColor={badgeColor} />
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
                        filterAlgorithm={filterAlgorithm}
                        absMax={absMax}
                        onPixelHover={handleHover}
                        onPixelLeave={handleLeave}
                        onPixelClick={clickAction?.type === 'callback' ? handlePixelClick : undefined}
                        overlayAlgorithm={overlayAlgorithm}
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

    if (clickAction?.type === 'link') {
        return (
            <Link to={clickAction.link} style={{ textDecoration: 'none', color: 'inherit', display: 'block', width: '100%', height: '100%' }}>
                {content}
            </Link>
        );
    }

    return content;
}