import { createContext, useContext, useState, type ReactNode } from 'react';
import { type ColormapName } from '../utils/colormaps';

interface ColormapContextValue {
  colormap: ColormapName;
  setColormap: (c: ColormapName) => void;
}

const ColormapContext = createContext<ColormapContextValue>({
  colormap: 'rd_bk_gn',
  setColormap: () => {},
});

export const ColormapProvider = ({ children }: { children: ReactNode }) => {
  const [colormap, setColormap] = useState<ColormapName>('rd_bk_gn');
  return (
    <ColormapContext.Provider value={{ colormap, setColormap }}>
      {children}
    </ColormapContext.Provider>
  );
};

export const useColormap = () => useContext(ColormapContext);
