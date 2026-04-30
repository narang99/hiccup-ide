import type { NodeFetchers } from "../../fetchers";
import { ActivationDisplay } from "../ActivationDisplay";

interface ConvOutActNodeProps {
    coordinate: string;
    kernelIdx?: number;
    fetchers?: NodeFetchers,
    maxSize?: number;
    className?: string; 
    title: string;
}

export default function ConvOutActNode(
    {kernelIdx, coordinate, fetchers, maxSize, className="", title}: ConvOutActNodeProps
) {
    return (
        <div className={`flex flex-col items-center gap-1 ${className}`}>
            {kernelIdx && <div className="font-bold text-xs">K{kernelIdx}</div>}
            {fetchers?.activation ? (
                <ActivationDisplay
                    coordinate={coordinate}
                    fetcher={fetchers.activation}
                    maxSize={maxSize}
                />
            ) : (
                <div className="w-10 h-10 bg-gray-400 rounded border" />
            )}
            <div className="text-xs text-gray-200">
                {title}
            </div>
            <div className="text-xs text-gray-300">🔍</div>
        </div>
    );
}