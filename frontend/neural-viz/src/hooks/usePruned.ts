import { create } from 'zustand';

interface PrunedStore {
  isPruned: boolean;
  graphAlias: string | null;
  setPruned: (isPruned: boolean, graphAlias?: string) => void;
  reset: () => void;
}

export const usePruned = create<PrunedStore>((set) => ({
  isPruned: false,
  graphAlias: null,
  setPruned: (isPruned: boolean, graphAlias?: string) => {
    set({ 
      isPruned, 
      graphAlias: isPruned ? (graphAlias || 'default_pruned_graph') : null 
    });
  },
  reset: () => set({ isPruned: false, graphAlias: null }),
}));