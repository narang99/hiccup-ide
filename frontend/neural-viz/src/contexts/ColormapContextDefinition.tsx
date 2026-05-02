import { createContext } from 'react';
import { type ColormapName } from '../utils/colormaps';

export type ScalingMode = 'global' | 'local';

interface ColormapContextValue {
  colormap: ColormapName;
  setColormap: (c: ColormapName) => void;
  scalingMode: ScalingMode;
  setScalingMode: (m: ScalingMode) => void;
}

export const ColormapContext = createContext<ColormapContextValue>({
  colormap: 'rd_bk_gn',
  setColormap: () => {},
  scalingMode: 'local',
  setScalingMode: () => {},
});