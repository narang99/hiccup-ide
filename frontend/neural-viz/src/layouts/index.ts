import type { Direction } from "../types/direction";
import type { LayerGroupLayout } from "./common";
import { makeEvenlySpacedHorizontalLayout } from "./horizontal";
import { makeEvenlySpacedVerticalLayout } from "./vertical";

export const makeEvenlySpacedLayout = (
  numChannels: number,
  childHeight: number,
  childWidth: number,
  padding: number,
  direction: Direction,
): LayerGroupLayout => {
    const directionFn = direction === "LR" ? makeEvenlySpacedHorizontalLayout : makeEvenlySpacedVerticalLayout;
    const layout = directionFn(numChannels, childHeight, childWidth, padding)
    return layout;
}