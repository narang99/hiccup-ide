import { create } from 'zustand';

interface LayerSettings {
  sliderValue: number;
}

interface LayerSettingsState {
  layerSettings: Record<string, LayerSettings>;
  getLayerSettings: (nodeId: string) => LayerSettings;
  updateSliderValue: (nodeId: string, value: number) => void;
  resetLayerSettings: (nodeId: string) => void;
  clearAllSettings: () => void;
}

const DEFAULT_LAYER_SETTINGS: LayerSettings = {
  sliderValue: 100,
};

export const useLayerSettingsStore = create<LayerSettingsState>((set, get) => ({
  layerSettings: {},
  
  getLayerSettings: (nodeId: string) => {
    const { layerSettings } = get();
    return layerSettings[nodeId] || DEFAULT_LAYER_SETTINGS;
  },
  
  updateSliderValue: (nodeId: string, value: number) => {
    set((state) => ({
      layerSettings: {
        ...state.layerSettings,
        [nodeId]: {
          ...state.layerSettings[nodeId],
          sliderValue: value,
        },
      },
    }));
  },
  
  resetLayerSettings: (nodeId: string) => {
    set((state) => ({
      layerSettings: {
        ...state.layerSettings,
        [nodeId]: DEFAULT_LAYER_SETTINGS,
      },
    }));
  },
  
  clearAllSettings: () => {
    set({ layerSettings: {} });
  },
}));