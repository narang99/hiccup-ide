import chroma from 'chroma-js';

export const COLORMAPS = {
  /** Red → near-black → neon green (dark mid, high contrast) */
  rd_bk_gn: chroma.scale(['#FF3131', '#333333', '#39FF14']).mode('lab'),
  /** Crimson → light grey → dark green (light mid) */
  rd_wht_gn: chroma.scale(['#B22222', '#D3D3D3', '#00A550']).mode('lab'),
} as const;

export type ColormapName = keyof typeof COLORMAPS;

/** Human-readable labels for the UI */
export const COLORMAP_META: Record<ColormapName, { label: string; gradient: string }> = {
  rd_bk_gn: {
    label: 'Dark',
    gradient: 'linear-gradient(to right, #FF3131, #333333, #39FF14)',
  },
  rd_wht_gn: {
    label: 'Light',
    gradient: 'linear-gradient(to right, #B22222, #D3D3D3, #00A550)',
  },
};

/**
 * Normalize a value symmetrically around 0.
 * Maps [-absMax, 0, +absMax] → [0, 0.5, 1].
 */
export function normalizeSymmetric(value: number, absMax: number): number {
  if (absMax === 0) return 0.5;
  return (value + absMax) / (2 * absMax);
}
