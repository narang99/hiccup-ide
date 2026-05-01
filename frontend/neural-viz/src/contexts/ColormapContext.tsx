import { useState, type ReactNode } from 'react';
import { type ColormapName } from '../utils/colormaps';
import { ColormapContext } from './ColormapContextDefinition';

export const ColormapProvider = ({ children }: { children: ReactNode }) => {
  const [colormap, setColormap] = useState<ColormapName>('rd_bk_gn');
  return (
    <ColormapContext.Provider value={{ colormap, setColormap }}>
      {children}
    </ColormapContext.Provider>
  );
};
