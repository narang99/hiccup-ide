import type { LayerGroupLayout } from "./common";

export const makeEvenlySpacedVerticalLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
): LayerGroupLayout => {
  // child node (H, W)
  // padding = p 
  // p + H + p + H + p + H + p
  // the amount of padding is simply H + 1
  // position of 0: p
  // 1: p + H + p
  // 2: p + H + p + H + p
  // total height = num_channels * child_height + (num_channels + 1) * padding
  const height = numChannels * childHeight + (numChannels + 1) * padding;
  const width = padding + childWidth;
  const parent = { height, width };

  const children = [];
  for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
    // position of 0: p
    // 1: p + H + p
    // 2: p + H + p + H + p
    const x = padding;
    const y = channelIndex * childHeight + (channelIndex + 1) * padding;
    children.push({ x: x, y: y });
  }

  return { parent, children };
}

export const makeGridVerticalLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
  rows: number = 10,
): LayerGroupLayout => {
  // Grid layout with fixed rows (vertical direction)
  // Calculate columns needed
  const cols = Math.ceil(numChannels / rows);
  
  // Calculate parent dimensions
  const width = cols * childWidth + (cols + 1) * padding;
  const height = rows * childHeight + (rows + 1) * padding;
  const parent = { height, width };

  const children = [];
  for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
    const col = Math.floor(channelIndex / rows);
    const row = channelIndex % rows;
    
    const x = col * childWidth + (col + 1) * padding;
    const y = row * childHeight + (row + 1) * padding;
    children.push({ x, y });
  }

  return { parent, children };
}