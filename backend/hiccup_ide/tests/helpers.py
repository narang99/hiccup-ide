import os
from django.core.files import File
from neural_data.models import Model, Input

# Hardcoded paths to real model data
MODEL_PT_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/model.pt"
INPUT_PT_PATH = "/Users/hariomnarang/Desktop/personal/hiccup-ide/pt-to-api/data/first-input-tens.pt"

def create_test_model_with_pt_file(alias="test-model", name="Test Model", definition=None):
    """Create a test model with pt_file if available, otherwise None"""
    if definition is None:
        definition = {}
    
    model = Model(
        alias=alias,
        name=name,
        definition=definition
    )
    
    # Only set pt_file if the file exists
    if os.path.exists(MODEL_PT_PATH):
        with open(MODEL_PT_PATH, 'rb') as f:
            model.pt_file.save('model.pt', File(f), save=False)
    
    model.save()
    return model

def create_test_input_with_pt_file(model, alias="test-input", name="Test Input", data_path="/path/to/data"):
    """Create a test input with pt_file if available, otherwise None"""
    input_obj = Input(
        model=model,
        alias=alias,
        name=name,
        data_path=data_path
    )
    
    # Only set pt_file if the file exists
    if os.path.exists(INPUT_PT_PATH):
        with open(INPUT_PT_PATH, 'rb') as f:
            input_obj.pt_file.save('input.pt', File(f), save=False)
    
    input_obj.save()
    return input_obj