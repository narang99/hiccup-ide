import json
from pathlib import Path
from typing import Dict, Union


def dump_coordinates_to_json_files(
    coordinate_data: Dict[str, Dict], 
    output_dir: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Common function to dump coordinate data to individual JSON files.
    
    Each coordinate gets its own JSON file named with the coordinate key.
    This is the shared functionality used by all data processors (activations, 
    contributions, weights) for consistent file output.
    
    Args:
        coordinate_data: Dict mapping coordinate strings to data dictionaries
        output_dir: Directory to save JSON files (created if doesn't exist)
        verbose: Whether to print file paths as they're saved
        
    File naming:
        - Coordinate "layers.0.out_2.in_1" -> "layers.0.out_2.in_1.json"
        - Coordinate "layers.0.out_2.bias" -> "layers.0.out_2.bias.json"
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for coordinate, data in coordinate_data.items():
        filepath = output_path / f"{coordinate}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if verbose:
            print(f"Saved: {filepath}")