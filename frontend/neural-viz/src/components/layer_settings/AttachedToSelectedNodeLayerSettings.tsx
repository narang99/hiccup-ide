import { useSelectedNodeStore } from "../../stores/selectedNodeStore"
import { LayerSettings } from "./LayerSettings"

export const AttachedToSelectedNodeLayerSettings = () => {
    const { selectedNode } = useSelectedNodeStore()
    return (
        <LayerSettings selectedNode={selectedNode} />
    )
}