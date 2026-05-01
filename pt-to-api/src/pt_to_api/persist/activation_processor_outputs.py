from pathlib import Path
from typing import Dict, Union

from .common import dump_coordinates_to_json_files


def dump_coordinates_to_files(coordinate_data: Dict[str, Dict], output_dir: Union[str, Path]) -> None:
    """
    Export activation coordinate data to individual JSON files for frontend consumption.
    Each coordinate gets its own JSON file named with the coordinate key.
    
    Args:
        coordinate_data: Output from activation_processor.process_activations_to_coordinates
        output_dir: Directory to write JSON files (created if doesn't exist)
    """
    dump_coordinates_to_json_files(coordinate_data, output_dir)