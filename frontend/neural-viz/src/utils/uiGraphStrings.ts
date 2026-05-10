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
    case "ReLUOutputPatchNode": {
      const miny = node.patch_min_y;
      const minx = node.patch_min_x;
      const maxy = node.patch_max_y;
      const maxx = node.patch_max_x;
      return `${node.type} ${node.layer_name}.out_${node.channel} (${miny},${minx}:${maxy},${maxx})`;
    }
    case "ModelInputPatchNode": {
      const miny = node.patch_min_y;
      const minx = node.patch_min_x;
      const maxy = node.patch_max_y;
      const maxx = node.patch_max_x;
      return `${node.type} ${node.layer_name}.out_${node.channel} (${miny},${minx}:${maxy},${maxx})`;
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
      return `UnknownNode ${JSON.stringify(node)}`;
  }
};

/**
 * Generates a descriptive text label for the UI node.
 */
export const getUIGraphNodeText = (node: Coordinate): string => {
  const prefix = `${node.type} (${node.layer_name})`;
  const id = getUIGraphNodeId(node);
  
  // If the ID already contains the descriptive info, we can just use it or a variant
  if (node.type === "Conv2dInputPatchNode" || node.type === "ReLUOutputPatchNode" || node.type === "ModelInputPatchNode") {
    return id;
  }
  
  return `${prefix} ${id}`;
};

/**
 * Generates the standardized coordinate string used for fetching activation data.
 * Mirrors the backend's to_coord_str function.
 */
export const getUIGraphNodeCoordinate = (node: Coordinate): string => {
  switch (node.type) {
    case "Conv2dInputCoordinate":
    case "Conv2dOutputCoordinate":
    case "ReLUInputCoordinate":
    case "ReLUOutputCoordinate":
    case "ModelInputCoordinate":
      return `${node.layer_name}.out_${node.channel}`;
    case "Conv2dSliceCoordinate":
      return `${node.layer_name}.out_${node.out_channel}.in_${node.in_channel}`;
    case "Conv2dInputPatchNode":
      return `${node.layer_name}.patch.in_${node.in_channel}.out_${node.out_channel}`;
    case "ReLUOutputPatchNode":
    case "ModelInputPatchNode":
      return `${node.layer_name}.patch.out_${node.channel}`;
    default:
      return "unknown.coordinate";
  }
};
