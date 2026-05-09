/**
 * Types for coordinating groups and pixels inside the network
 * 
 * NOTE: These types must be kept in sync with backend types in:
 * backend/hiccup_ide/neural_data/types.py
 */

export interface ImmutableModel {
  readonly [key: string]: unknown;
}

export interface Conv2dSliceGroup extends ImmutableModel {
  readonly type: "Conv2dSliceGroup";
  readonly layer_name: string;
  readonly layer_type: "conv2d";
  readonly coordinate_type: "slice";
  readonly in_channel: number;
  readonly out_channel: number;
}

export interface Conv2dSliceCoordinate extends ImmutableModel {
  readonly type: "Conv2dSliceCoordinate";
  readonly layer_name: string;
  readonly layer_type: "conv2d";
  readonly coordinate_type: "slice";
  readonly in_channel: number;
  readonly out_channel: number;
  readonly y: number;
  readonly x: number;
}

export interface ChannelGroup {
  readonly layer_name: string;
  readonly channel: number;
}

export interface Conv2dOutputGroup extends ChannelGroup {
  readonly type: "Conv2dOutputGroup";
  readonly layer_type: "conv2d";
  readonly coordinate_type: "output";
}

export interface Conv2dOutputCoordinate extends ChannelGroup {
  readonly type: "Conv2dOutputCoordinate";
  readonly layer_type: "conv2d";
  readonly coordinate_type: "output";
  readonly y: number;
  readonly x: number;
}

export interface Conv2dInputGroup extends ChannelGroup {
  readonly type: "Conv2dInputGroup";
  readonly layer_type: "conv2d";
  readonly coordinate_type: "input";
}

export interface Conv2dInputCoordinate extends ChannelGroup {
  readonly type: "Conv2dInputCoordinate";
  readonly layer_type: "conv2d";
  readonly coordinate_type: "input";
  readonly y: number;
  readonly x: number;
}

export interface ReLUInputGroup extends ChannelGroup {
  readonly type: "ReLUInputGroup";
  readonly layer_type: "relu";
  readonly coordinate_type: "input";
}

export interface ReLUInputCoordinate extends ChannelGroup {
  readonly type: "ReLUInputCoordinate";
  readonly layer_type: "relu";
  readonly coordinate_type: "input";
  readonly y: number;
  readonly x: number;
}

export interface ReLUOutputGroup extends ChannelGroup {
  readonly type: "ReLUOutputGroup";
  readonly layer_type: "relu";
  readonly coordinate_type: "output";
}

export interface ReLUOutputCoordinate extends ChannelGroup {
  readonly type: "ReLUOutputCoordinate";
  readonly layer_type: "relu";
  readonly coordinate_type: "output";
  readonly y: number;
  readonly x: number;
}

export interface ModelInputGroup extends ChannelGroup {
  readonly type: "ModelInputGroup";
  readonly layer_type: "input";
}

export interface ModelInputCoordinate extends ChannelGroup {
  readonly type: "ModelInputCoordinate";
  readonly layer_type: "input";
  readonly y: number;
  readonly x: number;
}

export type Coordinate =
  | Conv2dInputCoordinate
  | Conv2dOutputCoordinate
  | Conv2dSliceCoordinate
  | ModelInputCoordinate
  | ReLUInputCoordinate
  | ReLUOutputCoordinate;

export type Group =
  | Conv2dInputGroup
  | Conv2dSliceGroup
  | Conv2dOutputGroup
  | ModelInputGroup
  | ReLUInputGroup
  | ReLUOutputGroup;