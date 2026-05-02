/// algorithm for thresholding saliency maps
/// this is the simplest one
/// it basically says: dim every contrib lesser than the threshold
export interface SaliencyMapThresholdAlgorithm {
    type: "SaliencyMapThresholdAlgorithm"
    threshold: number;
}

/// discriminated type union possible
export type SalicenyMapPruneAlgorithm = SaliencyMapThresholdAlgorithm;
