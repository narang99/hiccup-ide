from pathlib import Path
import json

def dump_contribs_to_files(
    coord_data: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """
    Dump each coordinate entry as a separate JSON file, identical in
    structure to the activation JSON files the frontend already consumes.

    The file name is "<coordinate>.json", e.g.:
        layers.0.out_0.json
        layers.0.out_0.in_0.json
        inputs.out_0.json

    Parameters
    ----------
    coord_data : dict
        Output of process_contribs_to_coordinates.
    output_dir : path-like
        Directory to write JSON files into (created if it doesn't exist).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for coordinate, entry in coord_data.items():
        file_path = out_path / f"{coordinate}.json"
        with open(file_path, "w") as f:
            json.dump(entry, f, indent=2)