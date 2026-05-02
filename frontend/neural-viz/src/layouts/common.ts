import type { XYPosition } from "@xyflow/react";

export interface LayerGroupLayout {
  children: XYPosition[];
  parent: { height: number, width: number };
}
