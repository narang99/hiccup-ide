import json
from pathlib import Path
from typing import Dict


def dump_coordinates_to_files(coordinate_data: Dict[str, Dict], output_dir: str) -> None:
    """
    Export coordinate data to individual JSON files for frontend consumption.
    Each coordinate gets its own JSON file named with the coordinate key.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for coordinate, data in coordinate_data.items():
        filepath = output_path / f"{coordinate}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)