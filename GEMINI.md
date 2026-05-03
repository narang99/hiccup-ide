# Hiccup IDE - Claude Development Guide

## Project Architecture

This project consists of three main components:

### 1. `pt-to-api/` - PyTorch to JSON Converter
Converts PyTorch models and their activations/saliency data to JSON format that can be consumed by the backend and frontend.

**Purpose**: Transform PyTorch model artifacts into a standardized JSON format
**Tech Stack**: Python + PyTorch + uv package management

### 2. `backend/` - Django API Server  
Django application with Django Ninja API framework that serves model data.

**Purpose**: Serve JSON model data via REST API endpoints
**Tech Stack**: Django + Django Ninja + uv package management

### 3. `frontend/neural-viz/` - React Visualization Frontend
React application that visualizes neural network models and their activations.

**Purpose**: Interactive visualization of neural network models and data
**Tech Stack**: React + TypeScript + Vite

## Data Flow Workflow

1. **Generate JSON Data**: Use `pt-to-api/` to convert PyTorch models to JSON files
2. **Manual Data Loading**: Copy generated JSON files to appropriate backend location
3. **Backend Loading**: Manually load JSON data into Django database using management commands
4. **Frontend Consumption**: Frontend fetches data from backend REST API endpoints

### 4. Backend Conventions (Django)
- **Primary Key Access**: Always use the `.pk` attribute to access a model's primary key instead of `.id`. The linter is configured to prefer `.pk` for consistency and to avoid potential issues with custom primary key fields.

All Python components use `uv` for package management:

```bash
# In any Python project folder (pt-to-api/ or backend/)
uv sync                    # Install dependencies
uv run [command]          # Run commands in virtual environment
```

## Quick Start

1. **Setup pt-to-api**: `cd pt-to-api && uv sync`
2. **Setup backend**: `cd backend && uv sync`  
3. **Setup frontend**: `cd frontend/neural-viz && npm install`

Each component has its own `CLAUDE.md` with detailed development instructions.