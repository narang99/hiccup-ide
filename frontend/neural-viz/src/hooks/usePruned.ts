import { create } from 'zustand';

const PRUNED_STORAGE_KEY = 'hiccup-ide-pruned';

const getStoredPruned = (): boolean => {
  try {
    const stored = localStorage.getItem(PRUNED_STORAGE_KEY);
    if (stored === 'true' || stored === 'false') {
      return stored === 'true';
    }
  } catch {
    // localStorage not available or error reading
  }
  return false;
};

interface PrunedStore {
  isPruned: boolean;
  setPruned: (isPruned: boolean) => void;
  reset: () => void;
}

export const usePruned = create<PrunedStore>((set) => ({
  isPruned: getStoredPruned(),
  setPruned: (isPruned: boolean) => {
    try {
      localStorage.setItem(PRUNED_STORAGE_KEY, String(isPruned));
    } catch {
      // localStorage not available or quota exceeded
    }
    set({ isPruned });
  },
  reset: () => {
    try {
      localStorage.setItem(PRUNED_STORAGE_KEY, 'false');
    } catch {
      // localStorage not available or quota exceeded
    }
    set({ isPruned: false });
  },
}));