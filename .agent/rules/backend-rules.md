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