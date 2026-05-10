import type { Coordinate } from '../types/coordinates';

/**
 * Parses flat URL query parameters into a strongly-typed Coordinate object.
 * Returns a Coordinate object on success, or a string describing the error on failure.
 */
export function parseCoordinateFromURL(searchParams: URLSearchParams): Coordinate | string {
  const type = searchParams.get('type');
  if (!type) {
    return "Missing 'type' query parameter.";
  }

  const layer_name = searchParams.get('layer_name');
  if (!layer_name) {
    return "Missing 'layer_name' query parameter.";
  }

  const getNum = (key: string): number | null => {
    const val = searchParams.get(key);
    if (val === null) return null;
    const num = parseInt(val, 10);
    return isNaN(num) ? null : num;
  };

  const requireNum = (key: string): number | string => {
    const num = getNum(key);
    if (num === null) return `Missing or invalid numeric parameter: '${key}'`;
    return num;
  };

  switch (type) {
    case 'Conv2dSliceCoordinate': {
      const in_channel = requireNum('in_channel');
      const out_channel = requireNum('out_channel');
      const y = requireNum('y');
      const x = requireNum('x');
      const layer_type = searchParams.get('layer_type');
      const coordinate_type = searchParams.get('coordinate_type');

      if (typeof in_channel === 'string') return in_channel;
      if (typeof out_channel === 'string') return out_channel;
      if (typeof y === 'string') return y;
      if (typeof x === 'string') return x;
      if (layer_type !== 'conv2d') return "Invalid layer_type for Conv2dSliceCoordinate. Expected 'conv2d'.";
      if (coordinate_type !== 'slice') return "Invalid coordinate_type for Conv2dSliceCoordinate. Expected 'slice'.";

      return {
        type: 'Conv2dSliceCoordinate',
        layer_name,
        layer_type: 'conv2d',
        coordinate_type: 'slice',
        in_channel,
        out_channel,
        y,
        x,
      };
    }

    case 'Conv2dOutputCoordinate': {
      const channel = requireNum('channel');
      const y = requireNum('y');
      const x = requireNum('x');
      const layer_type = searchParams.get('layer_type');
      const coordinate_type = searchParams.get('coordinate_type');

      if (typeof channel === 'string') return channel;
      if (typeof y === 'string') return y;
      if (typeof x === 'string') return x;
      if (layer_type !== 'conv2d') return "Invalid layer_type for Conv2dOutputCoordinate. Expected 'conv2d'.";
      if (coordinate_type !== 'output') return "Invalid coordinate_type for Conv2dOutputCoordinate. Expected 'output'.";

      return {
        type: 'Conv2dOutputCoordinate',
        layer_name,
        layer_type: 'conv2d',
        coordinate_type: 'output',
        channel,
        y,
        x,
      };
    }

    case 'ReLUOutputCoordinate': {
      const channel = requireNum('channel');
      const y = requireNum('y');
      const x = requireNum('x');
      const layer_type = searchParams.get('layer_type');
      const coordinate_type = searchParams.get('coordinate_type');

      if (typeof channel === 'string') return channel;
      if (typeof y === 'string') return y;
      if (typeof x === 'string') return x;
      if (layer_type !== 'relu') return "Invalid layer_type for ReLUOutputCoordinate. Expected 'relu'.";
      if (coordinate_type !== 'output') return "Invalid coordinate_type for ReLUOutputCoordinate. Expected 'output'.";

      return {
        type: 'ReLUOutputCoordinate',
        layer_name,
        layer_type: 'relu',
        coordinate_type: 'output',
        channel,
        y,
        x,
      };
    }

    case 'ModelInputCoordinate': {
      const channel = requireNum('channel');
      const y = requireNum('y');
      const x = requireNum('x');
      const layer_type = searchParams.get('layer_type');

      if (typeof channel === 'string') return channel;
      if (typeof y === 'string') return y;
      if (typeof x === 'string') return x;
      if (layer_type !== 'input') return "Invalid layer_type for ModelInputCoordinate. Expected 'input'.";

      return {
        type: 'ModelInputCoordinate',
        layer_name,
        layer_type: 'input',
        channel,
        y,
        x,
      };
    }

    default:
      return `Unsupported coordinate type: '${type}'. Supported types: Conv2dSliceCoordinate, Conv2dOutputCoordinate, ReLUOutputCoordinate, ModelInputCoordinate.`;
  }
}
