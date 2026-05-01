import { type Node } from '@xyflow/react';
import { type ModelNode } from '../types/model';
import { createOutputKernelNode } from './kernelNodes';
import { ReluChannelNodeData } from '../components/nodes/ReluChannelNode';
import { DEFAULT_FETCHERS, type FetcherType } from '../fetchers';

export const createConv2dLayer = (
  modelNode: ModelNode,
  basePosition: { x: number; y: number },
  fetcherType: FetcherType
): Node[] => {
  const nodes: Node[] = [];
  const outChannels = modelNode.params.out_channels as number;
  const childWidth = 120;
  const childHeight = 120;
  const padding = 10;
  const ourWidth = outChannels * childWidth + (outChannels + 1) * padding;
  const ourHeight = padding + childHeight;

  // Step 1: Create all child nodes first to get their actual sizes

  // Step 5: Create parent layer node
  const parentLayerNode = {
    id: modelNode.id,
    type: 'LayerNode',
    position: basePosition,
    data: {
      label: `Conv2d Layer (${outChannels} kernels)`,
      layerType: 'Conv2d' as const,
      nodeCount: outChannels,
    },
    width: ourWidth,
    height: ourHeight,
  };
  nodes.push(parentLayerNode);

  for (let kernelIndex = 0; kernelIndex < outChannels; kernelIndex++) {
    const x = kernelIndex * childWidth + (kernelIndex + 1) * padding;
    const y = padding;
    const kernelNode = createOutputKernelNode(
      modelNode,
      kernelIndex,
      childHeight,
      childWidth,
      { x: x, y: y },
      DEFAULT_FETCHERS,
      fetcherType
    );
    nodes.push(kernelNode);
  }
  // Step 6: Position child nodes in grid and add to nodes array
  // childNodes.forEach((kernelNode, kernelIndex) => {
  //   const row = Math.floor(kernelIndex / kernelsPerRow);
  //   const col = kernelIndex % kernelsPerRow;

  //   // Update position with actual grid coordinates
  //   kernelNode.position = {
  //     x: padding + col * (actualChildWidth + padding),
  //     y: padding + row * (actualChildHeight + padding)
  //   };
  //   kernelNode.parentId = modelNode.id;
  //   kernelNode.extent = 'parent';
  //   nodes.push(kernelNode);
  // });

  return nodes;
};

export const createReLULayer = (
  modelNode: ModelNode,
  basePosition: { x: number; y: number },
  fetcherType: FetcherType
): Node[] => {
  const nodes: Node[] = [];
  // Get number of channels from the shape (assuming format [batch, channels, height, width])
  const numChannels = modelNode.shape.length > 1 ? modelNode.shape[1] : 1;
  const childWidth = 120;
  const childHeight = 120;
  const padding = 10;
  // child node (H, W)
  // padding = p 
  // p + W + p + W + p + W + p
  // the amount of padding is simply W + 1
  // for now i only do horizontal, although a clean grid layout is very useful but its okay
  // position of 0: p
  // 1: p + W + p
  // 2: p + W + p + W + p
  // total width = num_channels * child_width + (num_channels + 1) * padding
  const ourWidth = numChannels * childWidth + (numChannels + 1) * padding;
  const ourHeight = padding + childHeight;

  // Create parent layer node
  const parentLayerNode = {
    id: modelNode.id,
    type: 'LayerNode',
    position: basePosition,
    width: ourWidth,
    height: ourHeight,
    data: {
      label: `ReLU Layer (${numChannels} channels)`,
      layerType: 'ReLU' as const,
      nodeCount: numChannels,
    },
  };
  nodes.push(parentLayerNode);

  // Create channel sub-nodes with parentId in grid layout
  for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
    // const row = Math.floor(channelIndex / channelsPerRow);
    // const col = channelIndex % channelsPerRow;

    // position of 0: p
    // 1: p + W + p
    // 2: p + W + p + W + p

    const x = channelIndex * childWidth + (channelIndex + 1) * padding;
    const y = padding;

    const reluChannelNode: Node = {
      id: `${modelNode.id}-channel-${channelIndex}`,
      type: 'default',
      position: {
        x: x, y: y
      },
      data: {
        label: ReluChannelNodeData({ channelIndex, layerId: modelNode.id, fetchers: DEFAULT_FETCHERS, fetcherType }),
      },
      width: childWidth,
      height: childHeight,
      style: {
        background: 'transparent',
        border: '1px solid rgba(251, 191, 36, 0.35)',
        borderRadius: '6px',
        padding: 0,
        fontSize: '10px',
        overflow: 'hidden',
      },
      parentId: modelNode.id,
      extent: 'parent',
    };
    nodes.push(reluChannelNode);
  }

  return nodes;
};

export const createOtherLayer = (
  modelNode: ModelNode,
  basePosition: { x: number; y: number }
): Node[] => {
  const layerNode = {
    id: modelNode.id,
    type: 'LayerNode',
    position: basePosition,
    data: {
      label: modelNode.type,
      layerType: modelNode.type as 'Linear' | 'Flatten' | 'Input' | 'Output',
    },
    style: {
      width: 150,
      height: 100,
    },
  };

  return [layerNode];
};