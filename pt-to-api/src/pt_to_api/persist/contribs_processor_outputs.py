from pathlib import Path
from typing import Dict, Union

from .common import dump_coordinates_to_json_files


def dump_contribs_to_files(
    coord_data: Dict[str, Dict],
    output_dir: Union[str, Path],
) -> None:
    """
    Dump contribution coordinate data to individual JSON files.
    
    Creates one JSON file per coordinate, identical in structure to the 
    activation JSON files the frontend already consumes.

    File examples:
        - layers.0.out_0.json
        - layers.0.out_0.in_0.json
        - inputs.out_0.json

    Args:
        coord_data: Output from contrib_processor.process_contribs_to_coordinates
        output_dir: Directory to write JSON files (created if doesn't exist)
    """
    dump_coordinates_to_json_files(coord_data, output_dir)