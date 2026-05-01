import { useContext } from 'react';
import { ColormapContext } from '../contexts/ColormapContextDefinition';

export const useColormap = () => useContext(ColormapContext);