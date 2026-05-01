from pathlib import Path
from typing import Dict, Union

from .common import dump_coordinates_to_json_files


def dump_weights_to_files(
    coord_data: Dict[str, Dict],
    output_dir: Union[str, Path],
) -> None:
    """
    Dump model weight and bias coordinate data to individual JSON files.
    
    Creates one JSON file per coordinate, following the same structure as
    activation and contribution outputs for frontend consumption.
    
    File examples:
        - layers.0.out_0.in_0.json (kernel weights)
        - layers.0.out_0.bias.json (bias values)
        - layers.2.out_5.in_3.json (kernel weights)
        - layers.2.out_5.bias.json (bias values)
    
    Args:
        coord_data: Output from weights_processor.process_model_weights_to_coordinates
        output_dir: Directory to write JSON files (created if doesn't exist)
    """
    dump_coordinates_to_json_files(coord_data, output_dir)