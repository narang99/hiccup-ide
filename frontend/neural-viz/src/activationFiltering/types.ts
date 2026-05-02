/// algorithm for thresholding saliency maps
/// this is the simplest one
/// it basically says: dim every contrib lesser than the threshold
export interface ThresholdAlgorithm {
    type: "ThresholdAlgorithm";
    threshold: number;
}

export interface Id {
    type: "Id";
}

/// discriminated type union possible
export type ActivationFilterAlgorithm = ThresholdAlgorithm | Id;
