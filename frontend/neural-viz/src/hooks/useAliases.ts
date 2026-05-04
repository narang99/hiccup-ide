import { useContext } from 'react';
import { AliasContext } from '../contexts/AliasContextDefinition';

export const useAliases = () => {
  const context = useContext(AliasContext);
  if (!context) {
    throw new Error('useAliases must be used within an AliasProvider');
  }
  return context;
};
