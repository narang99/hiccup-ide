import { useSelectedNodeStore } from "../../stores/selectedNodeStore"
import { type ActivationFilterAlgorithm } from "../../activationFiltering/types";
import { type LayerThreshold } from "../../fetchers/threshold";
import { LayerSettings } from "./LayerSettings"

interface AttachedToSelectedNodeLayerSettingsProps {
    onChangeThreshold?: (layerId: string, sliderValue: number, algorithm: ActivationFilterAlgorithm) => void;
    onLoadInitialThresholds?: () => Promise<LayerThreshold[]>;
}

export const AttachedToSelectedNodeLayerSettings = ({ onChangeThreshold, onLoadInitialThresholds }: AttachedToSelectedNodeLayerSettingsProps) => {
    const { selectedNode } = useSelectedNodeStore()
    return (
        <LayerSettings 
            selectedNode={selectedNode} 
            onChangeThreshold={onChangeThreshold}
            onLoadInitialThresholds={onLoadInitialThresholds}
        />
    )
}