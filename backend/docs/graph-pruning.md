# Graph Pruning Flow

This document describes the conceptual flow and API interactions for the neural network graph pruning feature.

## Conceptual Overview

Pruning is a multi-step, sequential process where a user modifies saliency data for a specific layer, and the backend calculates the "ripple effect" on all upstream layers using re-propagation.

### 1. The Scratchpad Pattern
To allow for interactive experimentation without modifying base data or committed work immediately, the system uses a **Scratchpad** (`TempPruneSaliencyMap`). 
- When a session starts, all base data is cloned to the scratchpad.
- All subsequent modifications and re-propagation results are stored in the scratchpad.
- The session is only "permanent" once finalized.

### 2. Sequential Requirement
Pruning must follow a specific layer order (typically from the output towards the input). The status API tracks which layers have been processed to ensure the user follows this sequence.

---

## API Workflow

### Phase 1: Initialization
The frontend starts a new pruning session.
- **Route**: `POST /models/{m}/inputs/{i}/workflows/{w}/graphs/{g}/start_pruning/`
- **Action**: Backend deletes any existing scratchpad data for this graph and clones all `SaliencyMap` entries into `TempPruneSaliencyMap`.

### Phase 2: Status Checking
The frontend checks the current progress.
- **Route**: `GET /models/{m}/inputs/{i}/workflows/{w}/graphs/{g}/status/`
- **Action**: Returns a list of `done` layers and the `total` layer order. A layer is "done" if it has been modified in the current session (or was already finalized in a previous session).

### Phase 3: Pruning a Layer (The Core)
The frontend sends pruning instructions for a specific layer.
- **Route**: `POST /models/{m}/inputs/{i}/workflows/{w}/graphs/{g}/saliency_maps/`
- **Payload**: List of coordinates and the algorithm to apply (e.g., `ThresholdAlgorithm`).
- **Logic**:
    1. **Update Layer**: Backend applies the algorithm to the requested coordinates in the scratchpad and marks them as `is_modified`.
    2. **Tensor Reconstruction**: Backend gathers all output channel data for that layer from the scratchpad to form a PyTorch tensor.
    3. **Re-propagation**: Backend runs the model's contribution logic (via `pt-to-api` helpers) starting from the modified tensor.
    4. **Upstream Update**: All layers "upstream" (closer to the input) in the sequence are updated in the scratchpad with the new projected saliency values.

### Phase 4: Finalization
The user commits their changes.
- **Route**: `POST /models/{m}/inputs/{i}/workflows/{w}/graphs/{g}/finalize_pruning/`
- **Action**: Backend moves all `is_modified` entries from `TempPruneSaliencyMap` to permanent `WorkSaliencyMap` storage and clears the scratchpad.

---

## Technical Implementation Details

### Re-propagation Logic (Phase 3, Step 3)
The core pruning operation uses `get_contribs_for_inp_vectorized` from `pt-to-api` with a `LayerBackpropController`:

1. **Tensor Reconstruction**: Convert coordinate data back to tensor format `[1, C, H, W]` using `reconstruct_layer_tensor()`
2. **Contribution Calculation**: Run `get_contribs_for_inp_vectorized(batch_input, model, pruned_tensor, layer_name, device)`
3. **Layer Control**: `LayerBackpropController` determines which upstream layers get backpropagated vs set to zero tensors
4. **Coordinate Conversion**: Use `process_contribs_to_coordinates()` to convert tensors back to individual coordinate entries
5. **Bulk Update**: Update all upstream `TempPruneSaliencyMap` entries with new values

### Transaction Safety
- **Atomic Operations**: `create_batch_work_saliency_maps` wrapped in `@transaction.atomic`
- **Rollback Behavior**: If re-propagation fails, all database changes are rolled back
- **Consistency**: Scratchpad always remains in consistent state (either fully updated or unchanged)

### Performance Optimizations
- **Model Caching**: `@functools.lru_cache` caches loaded PyTorch model and input tensors
- **Bulk Database Operations**: Uses `bulk_update()` for upstream layer modifications
- **Coordinate Batching**: Processes multiple coordinates in single request

### Layer Ordering Enforcement
- **Fixed Order**: `["layers.3", "layers.2", "layers.1", "layers.0", "x"]` (output → input)
- **Sequential Validation**: Status API ensures layers completed in order
- **Upstream Definition**: Layer at index `i` affects all layers at indices `> i`

### Error Handling Strategy
- **Fail Fast**: Missing model/input files raise `FileNotFoundError` immediately
- **Validation**: All coordinates must belong to same layer, validated early
- **Context Preservation**: Exceptions re-raised with layer context using `raise ... from e`
- **No Silent Failures**: All failure modes now raise exceptions instead of logging and continuing

---

## Data Models Involved

- **`SaliencyMap`**: The "Gold Standard" base data. Never modified by pruning.
- **`TempPruneSaliencyMap`**: The active session scratchpad. Has `is_modified` flag to track user vs system changes.
- **`WorkSaliencyMap`**: Permanent storage for completed pruning work.
- **`WorkGraph`**: Contextual container for a specific pruning version/alias.
