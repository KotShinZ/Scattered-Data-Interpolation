# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure-Python implementation of 3D scattered data interpolation methods, designed to run without external dependencies (no NumPy, SciPy, or scikit-learn). The project compares ~26 different interpolation techniques including RBF, kriging variants, thin-plate splines, and Gaussian processes.

The implementation has two execution modes:
1. **CLI mode**: Run via `python src/compare_interpolation.py` to generate CSV results and SVG charts
2. **Browser mode**: Interactive visualization deployed to GitHub Pages using Pyodide

## Development Commands

### Running the CLI
```bash
python src/compare_interpolation.py
```
Generates:
- `data/dummy_3d_points.csv`: Synthetic test data (10 3D points with sin-based values + noise)
- `output/results_summary.csv`: RMSE and smoothness metrics for each method
- `output/interpolation_comparison.svg`: Bar chart visualization

### Running Tests
```bash
python -m pytest tests/
```

Tests are located in `tests/` and use the standard `unittest` framework.

### Running a Single Test
```bash
python -m pytest tests/test_line_predictions.py::LinePredictionSessionTests::test_line_predictions_use_custom_bounds
```

### Testing the Browser Version Locally
Open `docs/index.html` or `docs/line.html` directly in a browser. The Pyodide runtime will load `docs/compare_interpolation.py` in a Web Worker.

## Architecture

### Single-File Design

**All interpolation logic lives in `docs/compare_interpolation.py`** (~1800 lines). This file is shared between CLI and browser modes:

- **CLI wrapper**: `src/compare_interpolation.py` delegates to `docs/compare_interpolation.py` via `runpy`
- **Browser**: `docs/index.html` and `docs/line.html` load `docs/compare_interpolation.py` via Pyodide
- **Tests**: Import directly from `docs.compare_interpolation`

This architecture ensures the same interpolation code runs in both environments without duplication.

### Core Components in `docs/compare_interpolation.py`

1. **Linear algebra primitives** (lines ~36-98): Pure Python implementations of `dot`, `matmul`, `gaussian_elimination`, etc.

2. **Interpolator base class** (line ~243): All methods inherit from `Interpolator` with `fit(dataset)` and `predict(point)` interface

3. **26 interpolator implementations** (lines ~265-897):
   - Nearest neighbor and k-NN averaging
   - Inverse distance weighting
   - Global linear regression
   - RBF with 6 kernel types (linear, cubic, quintic, gaussian, multiquadric, inverse_multiquadric)
   - Thin-plate spline
   - Gaussian process regression
   - 17 kriging variants (ordinary, universal, exponential, gaussian, power, anisotropic, quadratic drift, linear, cubic, rational quadratic, hole-effect, Matérn 3/2, Matérn 5/2, logarithmic, Cauchy, stable, spline)

4. **Session-based API** (lines ~1513-1650):
   - `fit_session(dataset, ...)`: Train all interpolators and cache results in a `ComparisonSession`
   - `predict_session(slice_axis, slice_value, ...)`: Generate 2D plane predictions for visualization
   - `predict_line_session(varying_axis, fixed_values, ...)`: Generate 1D line predictions

5. **Evaluation metrics** (lines ~1238-1305):
   - RMSE on training points
   - First-order smoothness (gradient magnitude)
   - Second-order smoothness (Laplacian magnitude)

6. **SVG chart generation** (lines ~1325-1432): Pure Python string templating for bar charts

### Browser Integration

- `docs/index.html`: 3D surface visualization (fixes one axis, shows 2D plane)
- `docs/line.html`: 1D line plot (fixes two axes, shows variation along third)
- `docs/pyodide-worker.js`: Web Worker that runs Python code without blocking UI
- `docs/dataset-store.js`: localStorage wrapper for CSV persistence across page reloads

The browser UI:
1. Loads CSV or generates dummy data
2. Calls `fit_session()` in Pyodide to train all interpolators (cached)
3. On "Calculate Results" click, calls `predict_session()` or `predict_line_session()` with user-specified slice parameters
4. Renders results using Plotly.js for 3D surfaces or Chart.js for line plots

### Key Design Constraints

- **No external dependencies**: All matrix operations, distance calculations, and optimization are hand-rolled
- **Pyodide compatibility**: Code must run in browser via WASM (avoid platform-specific features)
- **Shared implementation**: `docs/compare_interpolation.py` is the single source of truth for both CLI and browser

## Working with Interpolators

### Adding a New Interpolator

1. Create a class inheriting from `Interpolator` or `BaseVariogramKriging` in `docs/compare_interpolation.py`
2. Implement `fit(self, dataset)` and `predict(self, point)` methods
3. Add instance to `create_interpolators()` list (line ~1173)
4. The new method will automatically appear in CLI output, test suite, and browser UI

### Modifying Existing Methods

All interpolator parameters are hardcoded in `create_interpolators()`. To change hyperparameters:
- Locate the interpolator instantiation (e.g., `GaussianProcessInterpolator(length_scale=0.35)`)
- Modify constructor arguments
- No changes needed elsewhere (session caching handles re-training)

## Common Pitfalls

### src vs docs Confusion

`src/compare_interpolation.py` is a 17-line wrapper. **Do not add interpolation logic there.**
All implementation goes in `docs/compare_interpolation.py`.

### Browser Session State

The browser caches `ComparisonSession` after calling `fit_session()`. Subsequent `predict_*_session()` calls reuse this cache. To force re-training, reload the page or upload new CSV data.

### Coordinate System

All data is 3D with `(x, y, z, value)` format:
- `(x, y, z)` are spatial coordinates
- `value` is the scalar being interpolated
- Interpolators predict `value` at query points `(x, y, z)`

### Plane/Line Slicing

- **Plane slice**: Fix one axis (e.g., `z=5.0`), predict on 2D grid of remaining axes
- **Line slice**: Fix two axes (e.g., `x=2.0, y=3.0`), predict along third axis

The UI resolution parameters control grid density but don't affect training.

## Data Flow

```
CSV input → fit_session() → ComparisonSession (all methods trained)
                                    ↓
                        predict_session(slice_axis, slice_value)
                                    ↓
                        2D grid predictions for each method
                                    ↓
                        SVG chart + metrics table
```

Browser adds an extra layer:
```
User uploads CSV → JS stores in localStorage → pyodide-worker.js
                                                        ↓
                                            Python fit_session() in Web Worker
                                                        ↓
                                            Cache session, return metrics
                                                        ↓
                        User clicks "Calculate" with slice params
                                                        ↓
                                            predict_session() → JSON
                                                        ↓
                                            Plotly.js renders surface
```

## File Synchronization

When updating `docs/compare_interpolation.py`, the changes automatically apply to:
- CLI execution (`src/compare_interpolation.py` delegates via `runpy`)
- Browser (loads `docs/compare_interpolation.py` from server)
- Tests (import from `docs.compare_interpolation`)

No manual synchronization needed between `src/` and `docs/`.
