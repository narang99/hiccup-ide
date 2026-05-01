import { createContext } from 'react';
import { type ColormapName } from '../utils/colormaps';

interface ColormapContextValue {
  colormap: ColormapName;
  setColormap: (c: ColormapName) => void;
}

export const ColormapContext = createContext<ColormapContextValue>({
  colormap: 'rd_bk_gn',
  setColormap: () => {},
});