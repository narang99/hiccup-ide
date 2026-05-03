import { useSelectedNodeStore } from "../../stores/selectedNodeStore"
import type { ActivationFilterAlgorithm } from "../../types/activationFiltering";
import { type LayerThreshold } from "../../types/threshold"
import { TopKSumSliderPreview } from "./TopKSumSliderPreview";

interface AttachedToSelectedNodeLayerSettingsProps {
    onChangeThreshold?: (layerId: string, sliderValue: number, algorithm: ActivationFilterAlgorithm) => void;
    onLoadInitialThresholds?: () => Promise<LayerThreshold[]>;
}

export const AttachedToSelectedNodeLayerSettings = ({ onChangeThreshold, onLoadInitialThresholds }: AttachedToSelectedNodeLayerSettingsProps) => {
    const { selectedNode } = useSelectedNodeStore()
    return (
        <TopKSumSliderPreview 
            selectedNode={selectedNode} 
            onChangeThreshold={onChangeThreshold}
            onLoadInitialThresholds={onLoadInitialThresholds}
        />
    )
}