/**
 * UI-specific graph types for neural network coordinate visualization.
 * 
 * NOTE: These types must be kept in sync with backend types in:
 * backend/hiccup_ide/neural_data/ui_graph_types.py
 */

import type {
  Conv2dInputCoordinate,
  Conv2dOutputCoordinate,
  ReLUInputCoordinate,
  ImmutableModel,
} from "./coordinates";

export type TuplifiedInputCoordinates = ReadonlyArray<
  readonly [Conv2dInputCoordinate, ReadonlyArray<UIGraphNode>]
>;

export interface Conv2dInputPatchNode extends ImmutableModel {
  readonly type: "Conv2dInputPatchNode";
  readonly layer_name: string;
  readonly layer_type: "conv2d";
  readonly coordinate_type: "input_patch";

  // The input channel this patch represents
  readonly in_channel: number;

  // The output channel this patch connects to
  readonly out_channel: number;

  // Patch boundaries (min/max coordinates of the receptive field)
  readonly patch_min_y: number;
  readonly patch_min_x: number;
  readonly patch_max_y: number;
  readonly patch_max_x: number;

  // References to the original Conv2dInputCoordinate objects
  // that form this patch (for detailed inspection if needed)
  readonly input_coordinates: TuplifiedInputCoordinates;
}

/**
 * UI Graph Node Types - Union of all node types used in UI graphs
 */
export type UIGraphNode =
  | Conv2dInputPatchNode
  | Conv2dOutputCoordinate
  | ReLUInputCoordinate;
