import { Handle, Position } from "@xyflow/react"
import type { Direction } from "../types/direction";

export interface SingleOrNoHandleOfDirectionProps {
    handleDirection: Direction | null;
    badgeColor: string;
}

export default function SingleOrNoHandle({
    handleDirection, badgeColor
}: SingleOrNoHandleOfDirectionProps) {
    const handleStyle = {
        background: badgeColor,
        border: '2px solid #fff',
        width: '8px',
        height: '8px',
    };
    return (
        <>
            {handleDirection === "TB" && (
                <>
                    <Handle type="target" position={Position.Top} style={handleStyle} />
                    <Handle type="source" position={Position.Bottom} style={handleStyle} />
                </>
            )}
            {
                handleDirection === "LR" && (
                    <>
                        <Handle type="target" position={Position.Left} style={handleStyle} />
                        <Handle type="source" position={Position.Right} style={handleStyle} />
                    </>
                )
            }
        </>
    )
}