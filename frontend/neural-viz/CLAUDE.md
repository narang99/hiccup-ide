# Frontend - Claude Development Guide

## Project Overview

React + TypeScript frontend for inspecting neural network models, their activations and saliency maps in detail.  

**Tech Stack:**
- React (UI framework)
- TypeScript (type safety)
- Vite (build tool)
- npm (package management)

## Development Commands

### Environment Setup
```bash
npm install          # Install dependencies
npm run dev          # Start development server
npm run build        # Build for production
```

### Code Quality
Use the npm scripts instead of running tools directly:

```bash
npm run lint         # ESLint + TypeScript checking
npm run lint:eslint  # ESLint only
npm run typecheck    # TypeScript checking only
```

**Important**: Always use `npm run lint` - do not use `eslint .` directly.

## Testing Guidelines

**Do not start servers for testing** - assume the user has already started any necessary development servers (frontend dev server, backend Django server). Focus on code implementation and validation through linting/typechecking only.

## React Query usage
Going forward, prefer using react query (also called tanstack query) for fetching data from backend.  
**DON'T CHANGE EXISTING CODE UNLESS EXPLICITLY ASKED TO REFACTOR**

## Data Loading Architecture

The frontend consumes model data via REST API calls to the Django backend:

- **Static JSON Development**: Currently uses static JSON files for development
- **API Integration**: Fetches data from backend REST endpoints  
- **Interface Consistency**: Same data loading interface regardless of backend

### Key Data Types
- Model definitions and architecture
- Activation data for specific coordinates
- Saliency maps and contribution data

## Architecture Documentation

### React Flow Implementation

Refer these docs for your specific tasks

- `docs/react-flow-architecture.md` - React Flow data model. You should read it, its generally useful
- `docs/creating-layers.md` - How layers are created and structured
- `docs/node-layouting.md` - Dagre-js automatic layout for layers and manual positioning for activations