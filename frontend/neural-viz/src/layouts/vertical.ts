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