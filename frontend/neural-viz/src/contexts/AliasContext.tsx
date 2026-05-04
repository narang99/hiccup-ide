import { type ReactNode, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { AliasContext } from './AliasContextDefinition';

export const AliasProvider = ({ children }: { children: ReactNode }) => {
  const { modelAlias, inputAlias, workAlias } = useParams();

  const value = useMemo(() => ({
    modelAlias: modelAlias || 'example-model',
    inputAlias: inputAlias || 'first-input',
    workAlias: workAlias || 'default-workflow',
  }), [modelAlias, inputAlias, workAlias]);

  return (
    <AliasContext.Provider value={value}>
      {children}
    </AliasContext.Provider>
  );
};
