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

### Project Structure
```
hiccup-ide/
├── src/
│   └── hiccup_ide/
│       ├── capture.py
│       ├── activation_processor.py
│       └── model_to_json.py
├── tests/
│   ├── test_capture.py
│   ├── test_activation_processor.py
│   └── conftest.py
├── pyproject.toml
├── uv.lock
└── CLAUDE.md
```