# Frontend - Claude Development Guide

## Project Overview

React + TypeScript frontend for visualizing neural network models and their activations/saliency data.

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