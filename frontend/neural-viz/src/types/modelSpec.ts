/**
 * Type definitions for neural network model specifications.
 * 
 * This module provides complete type definitions for the model definition JSON structure
 * using discriminated unions similar to the coordinate types in types.ts.
 * 
 * NOTE: These types must be kept in sync with backend types in:
 * backend/hiccup_ide/neural_data/model_spec.py
 */

export interface ImmutableModel {
  readonly [key: string]: unknown;
}

export interface BaseNode extends ImmutableModel {
  readonly id: string;
  readonly shape: number[]; // backwards compat, not useful
}

export interface InputParams extends ImmutableModel {
  readonly output_shape: number[]; // Output tensor shape (same as input shape)
}

export interface InputNode extends BaseNode {
  readonly type: "Input";
  readonly params: InputParams;
}

export interface Conv2dParams extends ImmutableModel {
  readonly in_channels: number;
  readonly out_channels: number;
  readonly kernel_size: number[]; // [height, width]
  readonly stride: number[]; // [height, width]
  readonly padding: number[]; // [height, width]
  readonly input_shape: number[]; // Input tensor shape [batch, channels, height, width]
  readonly output_shape: number[]; // Output tensor shape [batch, channels, height, width]
}

export interface Conv2dNode extends BaseNode {
  readonly type: "Conv2d";
  readonly params: Conv2dParams;
}

export interface ReLUParams extends ImmutableModel {
  readonly input_shape: number[]; // Input tensor shape
  readonly output_shape: number[]; // Output tensor shape (same as input for ReLU)
}

export interface ReLUNode extends BaseNode {
  readonly type: "ReLU";
  readonly params: ReLUParams;
}

export interface FlattenParams extends ImmutableModel {
  readonly input_shape: number[]; // Input tensor shape (e.g., [batch, channels, height, width])
  readonly output_shape: number[]; // Output tensor shape (e.g., [batch, flattened_size])
}

export interface FlattenNode extends BaseNode {
  readonly type: "Flatten";
  readonly params: FlattenParams;
}

export interface LinearParams extends ImmutableModel {
  readonly in_features: number;
  readonly out_features: number;
  readonly input_shape: number[]; // Input tensor shape
  readonly output_shape: number[]; // Output tensor shape
}

export interface LinearNode extends BaseNode {
  readonly type: "Linear";
  readonly params: LinearParams;
}

export interface OutputParams extends ImmutableModel {
  readonly input_shape: number[]; // Input tensor shape
}

export interface OutputNode extends BaseNode {
  readonly type: "Output";
  readonly params: OutputParams;
}

export interface ModelEdge extends ImmutableModel {
  readonly source: string;
  readonly target: string;
}

// Discriminated union of all layer node types
export type LayerNode =
  | InputNode
  | Conv2dNode
  | ReLUNode
  | FlattenNode
  | LinearNode
  | OutputNode;

export interface ModelDefinition extends ImmutableModel {
  readonly nodes: LayerNode[];
  readonly edges: ModelEdge[];
}