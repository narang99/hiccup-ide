import { createContext } from 'react';

export interface AliasContextValue {
  modelAlias: string;
  inputAlias: string;
  workAlias: string;
}

export const AliasContext = createContext<AliasContextValue>({
  modelAlias: 'example-model',
  inputAlias: 'first-input',
  workAlias: 'default-workflow',
});
