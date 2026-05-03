import type { ActivationFilterAlgorithm } from "./activationFiltering";

export interface LayerThreshold {
  id: number;
  layer_id: string;
  slider_value: number;
  algorithm: ActivationFilterAlgorithm;
}