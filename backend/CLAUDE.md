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