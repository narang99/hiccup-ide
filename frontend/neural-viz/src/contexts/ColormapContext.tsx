import { useState, type ReactNode } from 'react';
import { type ColormapName } from '../utils/colormaps';
import { ColormapContext, type ScalingMode } from './ColormapContextDefinition';

export const ColormapProvider = ({ children }: { children: ReactNode }) => {
  const [colormap, setColormap] = useState<ColormapName>('rd_bk_gn');
  const [scalingMode, setScalingMode] = useState<ScalingMode>('local');
  
  return (
    <ColormapContext.Provider value={{ colormap, setColormap, scalingMode, setScalingMode }}>
      {children}
    </ColormapContext.Provider>
  );
};
