import json
from pathlib import Path
from .model_loader import load_model
from .input_loader import load_inputs

def load_folder_data(folder_path, stdout):
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    meta_file = folder_path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found in {folder_path}")

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    model_info = meta_data.get("model")
    if not model_info:
        raise ValueError("meta.json missing 'model' information")

    model_obj, model_pt_path = load_model(model_info, folder_path, stdout)

    inputs_list = meta_data.get("inputs", [])
    load_inputs(inputs_list, model_obj, folder_path, model_pt_path, stdout)

    stdout.write("Folder loading complete!")
