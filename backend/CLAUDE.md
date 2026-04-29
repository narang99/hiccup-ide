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

# Run tests (when added)
uv run pytest
```

### Project Structure
```
hiccup-ide/
├── src/
│   └── hiccup_ide/
│       └── model_to_json.py
├── pyproject.toml
├── uv.lock
└── CLAUDE.md
```