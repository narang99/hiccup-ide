# Claude Development Guide

## Project Overview

This is a bare-bones Django application using Django Ninja for API development. Authentication is disabled for simplicity.

**Tech Stack:**
- Django (web framework)
- Django Ninja (API framework)
- uv (Python package manager)

## Environment Setup

This project uses `uv` for Python package management:

```bash
# Install dependencies
uv sync

# Run Django server
uv run manage.py runserver

# Run Django commands
uv run manage.py migrate
uv run manage.py shell
```

## URL Configuration Convention

**ALWAYS use trailing slashes in all URL paths** - this is a strict project convention.

### Correct Examples
```python
# urls.py
urlpatterns = [
    path("api/", api.urls),  # ✅ Correct
]

# API routes
@router.get("/models/{model_alias}/", response=ModelOut)  # ✅ Correct
```

### Incorrect Examples
```python
# DON'T DO THIS
path("api", api.urls)  # ❌ Missing trailing slash
@router.get("/models/{model_alias}", response=ModelOut)  # ❌ Missing trailing slash
```

This prevents 404 routing issues and ensures consistency across the API.

## Testing Guidelines

This project uses pytest with pytest-django for testing.

### Test Structure
- **Function-based tests only**: Use `def test_*()` functions, not test classes
- **Clear test names**: Use descriptive function names that explain what's being tested
- **Modular organization**: Separate test files by feature/module (e.g., `test_weights.py`, `test_models.py`)

### Development Workflow
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest hiccup_ide/tests/test_weights.py

# Run tests with verbose output
uv run pytest -v

# Run specific test function
uv run pytest hiccup_ide/tests/test_weights.py::test_create_weight_model
```

### Test File Naming
- Test files should start with `test_` (e.g., `test_weights.py`)
- Test functions should start with `test_` (e.g., `test_create_weight_model()`)

### Example Test Structure
```python
import pytest

@pytest.mark.django_db
def test_feature_with_valid_input():
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result

@pytest.mark.django_db 
def test_feature_with_edge_case():
    # Test edge cases, error conditions, etc.
    pass
```

**Important**: Always use function-based tests, never class-based tests. This keeps tests simple and focused.

## Code Organization Guidelines

### Function Size and Modularity
**Always keep functions small and readable**. When a function becomes large or complex:

1. **Break it down**: Extract pieces of logic into smaller sub-functions
2. **Single responsibility**: Each function should have one clear purpose
3. **Separate files for features**: If there are too many functions for a single feature, organize them in a separate module

### Example of Good Function Organization
```python
# Instead of one large function
def process_complex_data(data):
    # 50+ lines of mixed logic
    pass

# Break it down into focused functions
def validate_input_data(data):
    # 5-10 lines of validation logic
    pass

def transform_data(data):
    # 10-15 lines of transformation
    pass

def save_processed_data(data):
    # 5-10 lines of persistence logic
    pass

def process_complex_data(data):
    validated_data = validate_input_data(data)
    transformed_data = transform_data(validated_data)
    return save_processed_data(transformed_data)
```

## Documentation

- [Graph Pruning Flow](./docs/graph-pruning.md): Conceptual overview and API workflow for the graph pruning feature.