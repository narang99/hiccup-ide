import type { LayerGroupLayout } from "./common";

export const makeEvenlySpacedHorizontalLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
): LayerGroupLayout => {
  // child node (H, W)
  // padding = p 
  // p + W + p + W + p + W + p
  // the amount of padding is simply W + 1
  // for now i only do horizontal, although a clean grid layout is very useful but its okay
  // position of 0: p
  // 1: p + W + p
  // 2: p + W + p + W + p
  // total width = num_channels * child_width + (num_channels + 1) * padding
  const width = numChannels * childWidth + (numChannels + 1) * padding;
  const height = padding + childHeight;
  const parent = { height, width };

  const children = [];
  for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
    // position of 0: p
    // 1: p + W + p
    // 2: p + W + p + W + p
    const x = channelIndex * childWidth + (channelIndex + 1) * padding;
    const y = padding;
    children.push({ x: x, y: y });
  }

  return { parent, children };
}

export const makeGridHorizontalLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
  cols: number = 10,
): LayerGroupLayout => {
  // Grid layout with fixed columns (horizontal direction)
  // Calculate rows needed
  const rows = Math.ceil(numChannels / cols);
  
  // Calculate parent dimensions
  const width = cols * childWidth + (cols + 1) * padding;
  const height = rows * childHeight + (rows + 1) * padding;
  const parent = { height, width };

  const children = [];
  for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
    const row = Math.floor(channelIndex / cols);
    const col = channelIndex % cols;
    
    const x = col * childWidth + (col + 1) * padding;
    const y = row * childHeight + (row + 1) * padding;
    children.push({ x, y });
  }

  return { parent, children };
}