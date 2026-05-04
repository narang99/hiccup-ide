export type OverlayAlgorithm = 
    | { type: 'NoOverlay' }
    | { type: 'DrawRect', start: [number, number], end: [number, number], color?: string };
