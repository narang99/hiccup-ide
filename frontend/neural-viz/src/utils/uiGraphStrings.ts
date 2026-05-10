import type { Coordinate } from "../types/coordinates";

/**
 * Generates a unique string ID for a Coordinate node based on its type and properties.
 * This is used for both node IDs and edge source/target references.
 */
export const getUIGraphNodeId = (node: Coordinate): string => {
  switch (node.type) {
    case "Conv2dInputPatchNode": {
      const miny = node.patch_min_y;
      const minx = node.patch_min_x;
      const maxy = node.patch_max_y;
      const maxx = node.patch_max_x;
      return `${node.type} ${node.layer_name}.out_${node.out_channel}.in_${node.in_channel} (${miny},${minx}:${maxy},${maxx})`;
    }
    case "Conv2dOutputCoordinate":
      return `${node.type} ${node.layer_name}.out_${node.channel} (${node.y}, ${node.x})`;
    case "ReLUInputCoordinate":
      return `${node.type} ${node.layer_name}.in_${node.channel} (${node.y}, ${node.x})`;
    case "ReLUOutputCoordinate":
      return `${node.type} ${node.layer_name}.out_${node.channel} (${node.y}, ${node.x})`;
    case "Conv2dInputCoordinate":
      return `${node.type} ${node.layer_name}.in_${node.channel} (${node.y}, ${node.x})`;
    case "Conv2dSliceCoordinate":
      return `${node.type} ${node.layer_name}.out_${node.out_channel}.in_${node.in_channel} (${node.y}, ${node.x})`;
    case "ModelInputCoordinate":
      return `${node.type} ${node.layer_name}.out_${node.channel} (${node.y}, ${node.x})`;
    default:
      return `${(node as any).type} ${(node as any).layer_name} ${JSON.stringify(node)}`;
  }
};

/**
 * Generates a descriptive text label for the UI node.
 */
export const getUIGraphNodeText = (node: Coordinate): string => {
  const prefix = `${node.type} (${node.layer_name})`;
  const id = getUIGraphNodeId(node);
  
  // If the ID already contains the descriptive info, we can just use it or a variant
  if (node.type === "Conv2dInputPatchNode") {
    return id;
  }
  
  return `${prefix} ${id}`;
};
