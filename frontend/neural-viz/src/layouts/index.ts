import type { Direction } from "../types/direction";
import type { LayerGroupLayout } from "./common";
import { makeEvenlySpacedHorizontalLayout, makeGridHorizontalLayout } from "./horizontal";
import { makeEvenlySpacedVerticalLayout, makeGridVerticalLayout } from "./vertical";

export const makeEvenlySpacedLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
  direction: Direction,
  gridThreshold: number = 10,
): LayerGroupLayout => {
    if (numChannels > gridThreshold) {
        // Use grid layout when number of channels exceeds threshold
        const directionFn = direction === "LR" ? makeGridHorizontalLayout : makeGridVerticalLayout;
        const layout = directionFn(numChannels, childHeight, childWidth, padding)
        return layout;
    } else {
        // Use evenly spaced layout for smaller numbers of channels
        const directionFn = direction === "LR" ? makeEvenlySpacedHorizontalLayout : makeEvenlySpacedVerticalLayout;
        const layout = directionFn(numChannels, childHeight, childWidth, padding)
        return layout;
    }
}