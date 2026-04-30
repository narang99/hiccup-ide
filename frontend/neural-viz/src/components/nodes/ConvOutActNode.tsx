import type { NodeFetchers } from "../../fetchers";
import { ActivationDisplay } from "../ActivationDisplay";

interface ConvOutActNodeProps {
    coordinate: string;
    kernelIdx?: number;
    fetchers?: NodeFetchers;
    maxSize?: number;
    className?: string;
    title: string;
}

export default function ConvOutActNode(
    { kernelIdx, coordinate, fetchers, maxSize = 84, className = "", title }: ConvOutActNodeProps
) {
    return (
        <div className={className} style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>

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
                {kernelIdx !== undefined && (
                    <>
                        <span style={{
                            fontSize: 9,
                            fontWeight: 700,
                            letterSpacing: '0.06em',
                            textTransform: 'uppercase' as const,
                            color: '#60a5fa',
                            lineHeight: 1,
                        }}>
                            K{kernelIdx}
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
                {fetchers?.activation ? (
                    <ActivationDisplay
                        coordinate={coordinate}
                        fetcher={fetchers.activation}
                        maxSize={maxSize}
                    />
                ) : (
                    <div style={{
                        width: maxSize,
                        height: maxSize,
                        background: '#1e1e2e',
                        borderRadius: 4,
                        border: '1px solid rgba(255,255,255,0.08)',
                    }} />
                )}
            </div>

        </div>
    );
}