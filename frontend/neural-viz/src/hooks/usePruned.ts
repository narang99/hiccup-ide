import { create } from 'zustand';

interface PrunedStore {
  isPruned: boolean;
  setPruned: (isPruned: boolean) => void;
  reset: () => void;
}

export const usePruned = create<PrunedStore>((set) => ({
  isPruned: false,
  setPruned: (isPruned: boolean) => {
    set({ isPruned });
  },
  reset: () => set({ isPruned: false }),
}));