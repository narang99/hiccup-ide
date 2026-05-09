import os
from pathlib import Path
from django.core.files import File
from io import StringIO
from neural_data.models import Model, Input

# Base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Try to find real model data relative to project root, fallback to hardcoded absolute paths
DEFAULT_MODEL_PT_PATH = BASE_DIR / "pt-to-api" / "data" / "model.pt"
DEFAULT_INPUT_PT_PATH = BASE_DIR / "pt-to-api" / "data" / "first-input-tens.pt"

MODEL_PT_PATH = str(DEFAULT_MODEL_PT_PATH) if DEFAULT_MODEL_PT_PATH.exists() else "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/model.pt"
INPUT_PT_PATH = str(DEFAULT_INPUT_PT_PATH) if DEFAULT_INPUT_PT_PATH.exists() else "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/first-input-tens.pt"

def create_test_model_with_pt_file(alias="test-model", name="Test Model", definition=None):
    """Create a test model. Uses real model loader if pt file is available, otherwise manual creation."""
    if os.path.exists(MODEL_PT_PATH):
        from neural_data.load_folder.model_loader import load_model

        folder_path = Path(MODEL_PT_PATH).parent.parent
        model_info = {
            "name": alias,
            "path": "data/" + os.path.basename(MODEL_PT_PATH)
        }
        stdout = StringIO()
        model_obj, _ = load_model(model_info, folder_path, stdout)
        return model_obj

    if definition is None:
        definition = {}

    model = Model.objects.create(
        alias=alias,
        name=name,
        definition=definition
    )
    return model

def create_test_input_with_pt_file(model, alias="test-input", name="Test Input", data_path="/path/to/data"):
    """Create a test input. Uses real input loader if pt file is available, otherwise manual creation."""
    if os.path.exists(INPUT_PT_PATH):
        from neural_data.load_folder.input_loader import load_inputs

        folder_path = Path(INPUT_PT_PATH).parent.parent
        input_info = {
            "path": "data/" + os.path.basename(INPUT_PT_PATH),
            "label": 0 # Default to 0 for tests
        }
        stdout = StringIO()
        load_inputs([input_info], model, folder_path, model.pt_file.path, stdout)

        # Loader uses stem as alias
        input_alias = Path(INPUT_PT_PATH).stem
        return Input.objects.get(model=model, alias=input_alias)

    input_obj = Input.objects.create(
        model=model,
        alias=alias,
        name=name,
        data_path=data_path
    )
    return input_obj