export type OverlayAlgorithm = 
    | { type: 'NoOverlay' }
    | { type: 'DrawRect', rects: { start: [number, number], end: [number, number], color?: string }[] };
