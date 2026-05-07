import { type ModelData, type ModelNode } from '../types/model';
import { getSliceStatus } from './saliency_map';

export interface SliceStatusResult {
  coordinate: string;
  is_done: boolean;
}

export interface AggregateSliceStatusResult {
  coordinate: string;
  is_done: boolean;
  checked_coordinates: string[];
  completed_coordinates: string[];
  total_coordinates: number;
  completed_count: number;
}

/**
 * Get slice status for a single coordinate
 * This is a direct wrapper around the existing getSliceStatus function
 */
export async function getSingleSliceStatus(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  coordinate: string
): Promise<SliceStatusResult> {
  return getSliceStatus(modelAlias, inputAlias, workAlias, coordinate);
}

/**
 * Get aggregated slice status for a conv output channel
 * Checks all input slices for the given output channel coordinate
 * Format: <layer_name>.out_<channum>
 */
export async function getConvOutputChannelStatus(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  layerName: string,
  outputChannel: number,
  inputChannels: number
): Promise<AggregateSliceStatusResult> {
  
  // Generate all input slice coordinates for this output channel
  const inputSliceCoordinates = Array.from(
    { length: inputChannels }, 
    (_, inputIdx) => `${layerName}.out_${outputChannel}.in_${inputIdx}`
  );

  // Check status for all input slices
  const sliceStatuses = await Promise.all(
    inputSliceCoordinates.map(coord => 
      getSingleSliceStatus(modelAlias, inputAlias, workAlias, coord)
    )
  );

  // Determine overall status (all slices must be done for output channel to be done)
  const completedSlices = sliceStatuses.filter(status => status.is_done);
  const isAllDone = completedSlices.length === sliceStatuses.length;

  const outputChannelCoordinate = `${layerName}.out_${outputChannel}`;

  return {
    coordinate: outputChannelCoordinate,
    is_done: isAllDone,
    checked_coordinates: inputSliceCoordinates,
    completed_coordinates: completedSlices.map(status => status.coordinate),
    total_coordinates: inputSliceCoordinates.length,
    completed_count: completedSlices.length
  };
}

/**
 * Get slice status for a specific input slice of an output channel
 * Format: <layer_name>.out_<channum>.in_<channum>
 */
export async function getConvInputSliceStatus(
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  inputSliceCoordinate: string
): Promise<SliceStatusResult> {
  // Validate the input slice coordinate format
  const match = inputSliceCoordinate.match(/^(.+)\.out_(\d+)\.in_(\d+)$/);
  if (!match) {
    throw new Error(`Invalid input slice coordinate format: ${inputSliceCoordinate}`);
  }

  return getSingleSliceStatus(modelAlias, inputAlias, workAlias, inputSliceCoordinate);
}

/**
 * Helper function to extract input channels count from a layer node
 */
function getInputChannelsForLayer(layerNode: ModelNode): number {
  const { type, params } = layerNode;
  
  if (type === 'Conv2d' || type === 'Conv1d' || type === 'Conv3d') {
    const inChannels = params.in_channels;
    if (typeof inChannels === 'number') {
      return inChannels;
    }
  }
  
  throw new Error(`Unable to determine input channels for layer type: ${type}`);
}

