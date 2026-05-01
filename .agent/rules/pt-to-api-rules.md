# Claude Development Guide

## Python Environment Management with uv

This project uses `uv` for fast Python package management and virtual environment handling.


### Common Commands

#### Environment Management
```bash
# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Add new dependency
uv add torch torchvision

# Add development dependency
uv add --dev pytest black

# Remove dependency
uv remove package-name
```

#### Running Code
```bash
# Run Python scripts
uv run python src/hiccup_ide/model_to_json.py

# Run with specific Python version
uv run --python 3.11 python script.py
```

#### Development Workflow
```bash
# Install all dependencies including dev
uv sync --dev

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_activation_processor.py

# Run tests with verbose output
uv run pytest -v
```

## Testing Guidelines

This project uses pytest for testing with the following preferences:

### Test Structure
- **Function-based tests only**: Use `def test_*()` functions, not test classes unless absolutely necessary
- **Modular organization**: Separate test files by feature/module (e.g., `test_activation_processor.py`, `test_capture.py`)
- **Clear test names**: Use descriptive function names that explain what's being tested

### Development Workflow
1. **Adhoc scripts for development**: Feel free to create temporary test scripts while developing features
2. **Convert to pytest later**: Once a feature is working, convert your adhoc tests to proper pytest functions
3. **Keep tests simple**: Focus on testing behavior, not implementation details

### Test File Naming
- Test files should start with `test_` (e.g., `test_activation_processor.py`)
- Test functions should start with `test_` (e.g., `test_generate_coordinates()`)

### Example Test Structure
```python
def test_feature_with_valid_input():
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result

def test_feature_with_edge_case():
    # Test edge cases, error conditions, etc.
    pass
```

## Data Loading Interface Guidelines

### Backend Data Interfaces
- **Separation of concerns**: Keep data persistence/export functions in the `persist/` submodule
- **Minimal interfaces**: Only implement what's currently needed, avoid over-engineering
- **Clear boundaries**: Data export/import should have simple, focused functions

### Frontend Data Loading
- **Separate data loading files**: All data loading functionality should be in separate files with clear interfaces
- **Easy to change**: Data loading implementations should be easily swappable (static JSON files → REST API later)
- **Interface consistency**: Maintain the same interface regardless of backend implementation

### Current Implementation
- **Static JSON files**: Currently using individual JSON files per coordinate for frontend development
- **Future migration**: Will be replaced with REST API endpoints later, but interface stays the same

## PyTorch to JSON Conversion

This component converts PyTorch models and their data to JSON format for consumption by the backend and frontend.

### Purpose
- Convert PyTorch models to standardized JSON schema
- Extract activation data from model runs
- Generate saliency maps and contribution data
- Output JSON files that can be loaded into the backend database

### Typical Workflow
1. Run PyTorch model conversion scripts in `src/pt_to_api/`
2. Generate JSON files with model definitions, activations, and saliency data
3. Copy output JSON files to backend for manual loading
4. Backend imports this data into Django models

### Key Components
- **model_to_json.py**: Converts PyTorch model architecture to JSON schema
- **activation_processor.py**: Extracts and processes activation data
- **contrib_processor.py**: Generates saliency maps and attribution data
- **capture.py**: Hooks for capturing intermediate model outputs

### Project Structure
```
pt-to-api/
├── src/
│   └── pt_to_api/
│       ├── model_to_json.py
│       ├── activation_processor.py
│       ├── contrib_processor.py
│       ├── capture.py
│       └── utils.py
├── notebooks/
│   └── test-model-load.ipynb
├── pyproject.toml
├── uv.lock
└── CLAUDE.md
```

