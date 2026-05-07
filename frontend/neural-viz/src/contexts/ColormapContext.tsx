import { useState, useEffect, type ReactNode } from 'react';
import { type ColormapName, COLORMAPS } from '../utils/colormaps';
import { ColormapContext, type ScalingMode } from './ColormapContextDefinition';

const COLORMAP_STORAGE_KEY = 'hiccup-ide-colormap';
const SCALING_MODE_STORAGE_KEY = 'hiccup-ide-scaling-mode';

const getStoredColormap = (): ColormapName => {
  try {
    const stored = localStorage.getItem(COLORMAP_STORAGE_KEY);
    if (stored && stored in COLORMAPS) {
      return stored as ColormapName;
    }
  } catch {
    // localStorage not available or error reading
  }
  return 'rd_bk_gn';
};

const getStoredScalingMode = (): ScalingMode => {
  try {
    const stored = localStorage.getItem(SCALING_MODE_STORAGE_KEY);
    if (stored === 'global' || stored === 'local') {
      return stored;
    }
  } catch {
    // localStorage not available or error reading
  }
  return 'global';
};

export const ColormapProvider = ({ children }: { children: ReactNode }) => {
  const [colormap, setColormap] = useState<ColormapName>(getStoredColormap);
  const [scalingMode, setScalingMode] = useState<ScalingMode>(getStoredScalingMode);
  
  useEffect(() => {
    try {
      localStorage.setItem(COLORMAP_STORAGE_KEY, colormap);
    } catch {
      // localStorage not available or quota exceeded
    }
  }, [colormap]);
  
  useEffect(() => {
    try {
      localStorage.setItem(SCALING_MODE_STORAGE_KEY, scalingMode);
    } catch {
      // localStorage not available or quota exceeded
    }
  }, [scalingMode]);
  
  return (
    <ColormapContext.Provider value={{ colormap, setColormap, scalingMode, setScalingMode }}>
      {children}
    </ColormapContext.Provider>
  );
};
