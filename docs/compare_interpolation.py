#!/usr/bin/env python3
"""Compare 3D scattered data interpolation methods without external dependencies.

This script generates synthetic 3D scattered data, saves it as CSV, fits
multiple interpolators, evaluates their predictive accuracy and smoothness, and
produces an SVG summary chart.

All numerical routines are implemented using the Python standard library
because third-party scientific packages are unavailable in the execution
environment.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Point3D = Tuple[float, float, float]

AXIS_LABELS = ["X", "Y", "Z"]
# Line slices are visualized as dense curves, so oversample beyond the coarse
# cube grid we use for plane surfaces.
LINE_SLICE_RESOLUTION = 60


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python implementations)
# ---------------------------------------------------------------------------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    if not a or not b:
        return []
    assert len(a[0]) == len(b)
    b_t = transpose(b)
    return [[dot(row, col) for col in b_t] for row in a]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [dot(row, v) for row in a]


def identity(size: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def gaussian_elimination(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""

    n = len(a)
    # Augment matrix
    aug = [row[:] + [val] for row, val in zip(a, b)]

    for col in range(n):
        # Pivot selection
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        # Eliminate column in other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    return gaussian_elimination([row[:] for row in a], list(b))


def inverse(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    aug = [row[:] + identity_row[:] for row, identity_row in zip(a, identity(n))]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(col, 2 * n):
            aug[col][j] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, 2 * n):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def underlying_function(point: Point3D) -> float:
    x, y, z = point
    return (
        math.sin(2 * math.pi * x)
        + 0.6 * math.sin(2 * math.pi * y)
        + 0.3 * math.sin(2 * math.pi * z)
    )


def generate_dataset(num_points: int, seed: int = 42) -> List[Tuple[Point3D, float]]:
    random.seed(seed)
    if num_points <= 0:
        return []

    data = []
    max_index = max(1, num_points - 1)
    for index in range(num_points):
        t = index / max_index
        point = (
            t,
            0.5 + 0.5 * math.sin(2 * math.pi * t),
            0.5 + 0.5 * math.cos(2 * math.pi * t),
        )
        noise = random.uniform(-0.01, 0.01)
        value = underlying_function(point) + noise
        data.append((point, value))
    return data


def write_csv(path: str, data: List[Tuple[Point3D, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z", "value"])
        for (x, y, z), value in data:
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{value:.6f}"])


def compute_axis_bounds(points: Sequence[Point3D]) -> List[Tuple[float, float]]:
    if not points:
        return [(0.0, 1.0)] * 3

    xs, ys, zs = zip(*points)
    bounds = []
    for axis_values in (xs, ys, zs):
        minimum = min(axis_values)
        maximum = max(axis_values)
        if math.isclose(minimum, maximum):
            maximum = minimum + 1.0
        bounds.append((minimum, maximum))
    return bounds


def normalize_dataset(dataset_input: Any) -> List[Tuple[Point3D, float]]:
    """Coerce various dataset representations into ``[(Point3D, value), ...]``."""

    if dataset_input is None:
        raise ValueError("Dataset input is None.")

    def _coerce_point(value: Sequence[float]) -> Point3D:
        if len(value) != 3:
            raise ValueError("Each point must contain exactly three coordinates.")
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError("Point coordinates must be numeric.") from exc

    normalized: List[Tuple[Point3D, float]] = []

    if isinstance(dataset_input, dict):
        points = dataset_input.get("points")
        values = dataset_input.get("values")
        if points is None or values is None:
            raise ValueError("Dictionary dataset must include 'points' and 'values'.")
        if len(points) != len(values):
            raise ValueError("'points' and 'values' must have the same length.")
        iterator = zip(points, values)
    else:
        iterator = dataset_input

    for item in iterator:
        if isinstance(item, dict):
            if not {"x", "y", "z", "value"}.issubset(item.keys()):
                raise ValueError("Dictionary rows must include x, y, z, and value keys.")
            point = _coerce_point((item["x"], item["y"], item["z"]))
            try:
                value = float(item["value"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Values must be numeric.") from exc
        elif isinstance(item, (list, tuple)):
            if len(item) == 2 and isinstance(item[0], (list, tuple)):
                point = _coerce_point(item[0])
                try:
                    value = float(item[1])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Values must be numeric.") from exc
            elif len(item) == 4:
                point = _coerce_point(item[:3])
                try:
                    value = float(item[3])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Values must be numeric.") from exc
            else:
                raise ValueError(
                    "Iterable rows must be ((x, y, z), value) or (x, y, z, value)."
                )
        else:
            raise ValueError(
                "Dataset rows must be dicts or sequences describing a point and value."
            )

        normalized.append((point, value))

    if not normalized:
        raise ValueError("Dataset must contain at least one row.")

    return normalized


# ---------------------------------------------------------------------------
# Interpolator implementations
# ---------------------------------------------------------------------------


class Interpolator:
    name: str
    smoothness_class: str

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        raise NotImplementedError

    def predict(self, point: Point3D) -> float:
        raise NotImplementedError


class NearestNeighborInterpolator(Interpolator):
    name = "Nearest Neighbor"
    smoothness_class = "C0 (discontinuous)"

    def __init__(self) -> None:
        self.points: List[Point3D] = []
        self.values: List[float] = []

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        self.values = list(values)

    def predict(self, point: Point3D) -> float:
        best_dist = float("inf")
        best_val = 0.0
        for p, v in zip(self.points, self.values):
            dist = math.dist(point, p)
            if dist < best_dist:
                best_dist = dist
                best_val = v
        return best_val


class KNNUniformInterpolator(Interpolator):
    name = "K-Neighbors (uniform)"
    smoothness_class = "C0 (piecewise constant)"

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self.points: List[Point3D] = []
        self.values: List[float] = []

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        self.values = list(values)

    def predict(self, point: Point3D) -> float:
        distances = [(math.dist(point, p), v) for p, v in zip(self.points, self.values)]
        distances.sort(key=lambda item: item[0])
        neighbors = distances[: self.k]
        if not neighbors:
            return 0.0
        return sum(v for _, v in neighbors) / len(neighbors)


class InverseDistanceWeightingInterpolator(Interpolator):
    name = "Inverse Distance Weighting"
    smoothness_class = "C0 (weighted average)"

    def __init__(self, power: float = 2.0, epsilon: float = 1e-6) -> None:
        self.power = power
        self.epsilon = epsilon
        self.points: List[Point3D] = []
        self.values: List[float] = []

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        self.values = list(values)

    def predict(self, point: Point3D) -> float:
        weights = []
        weighted_values = []
        for p, v in zip(self.points, self.values):
            dist = math.dist(point, p)
            if dist < self.epsilon:
                return v
            w = 1.0 / (dist ** self.power)
            weights.append(w)
            weighted_values.append(w * v)
        if not weights:
            return 0.0
        return sum(weighted_values) / sum(weights)


class LinearRegressionInterpolator(Interpolator):
    name = "Linear (global regression)"
    smoothness_class = "C1 (global polynomial)"

    def __init__(self) -> None:
        self.coefficients: List[float] = []

    def _design_row(self, point: Point3D) -> List[float]:
        x, y, z = point
        return [1.0, x, y, z, x * y, y * z, z * x]

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        design = [self._design_row(p) for p in points]
        design_t = transpose(design)
        normal_matrix = matmul(design_t, design)
        normal_rhs = matvec(design_t, values)
        self.coefficients = solve_linear_system(normal_matrix, normal_rhs)

    def predict(self, point: Point3D) -> float:
        if not self.coefficients:
            return 0.0
        row = self._design_row(point)
        return dot(self.coefficients, row)


# ---------------------------------------------------------------------------
# Radial basis function interpolators
# ---------------------------------------------------------------------------


KernelFunction = Callable[[float], float]


def rbf_kernel_factory(name: str, epsilon: float = 1.0) -> KernelFunction:
    if name == "gaussian":
        return lambda r: math.exp(-((epsilon * r) ** 2))
    if name == "multiquadric":
        return lambda r: math.sqrt(1.0 + (epsilon * r) ** 2)
    if name == "inverse_multiquadric":
        return lambda r: 1.0 / math.sqrt(1.0 + (epsilon * r) ** 2)
    if name == "cubic":
        return lambda r: r ** 3
    if name == "quintic":
        return lambda r: r ** 5
    if name == "linear":
        return lambda r: r
    if name == "thin_plate_spline":
        def kernel(r: float) -> float:
            if r == 0.0:
                return 0.0
            return (r ** 2) * math.log(r)
        return kernel
    raise ValueError(f"Unknown RBF kernel: {name}")


class RBFInterpolator(Interpolator):
    smoothness_class = "C1+ (radial basis)"

    def __init__(self, kernel_name: str, epsilon: float = 1.0, regularization: float = 1e-6) -> None:
        self.kernel_name = kernel_name
        self.kernel = rbf_kernel_factory(kernel_name, epsilon)
        self.regularization = regularization
        self.points: List[Point3D] = []
        self.weights: List[float] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"RBF ({self.kernel_name})"

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        n = len(self.points)
        if n == 0:
            self.weights = []
            return
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                r = math.dist(self.points[i], self.points[j])
                matrix[i][j] = self.kernel(r)
            matrix[i][i] += self.regularization
        self.weights = solve_linear_system(matrix, list(values))

    def predict(self, point: Point3D) -> float:
        if not self.points:
            return 0.0
        values = []
        for w, p in zip(self.weights, self.points):
            r = math.dist(point, p)
            values.append(w * self.kernel(r))
        return sum(values)


class ThinPlateSplineInterpolator(RBFInterpolator):
    """Thin-plate spline specialization with a user-friendly label."""

    name = "Thin-Plate Spline"
    smoothness_class = "C1 (thin-plate spline)"

    def __init__(self, regularization: float = 1e-6) -> None:
        super().__init__("thin_plate_spline", regularization=regularization)


# ---------------------------------------------------------------------------
# Gaussian process and kriging-style interpolators
# ---------------------------------------------------------------------------


class GaussianProcessInterpolator(Interpolator):
    name = "Gaussian Process Regression"
    smoothness_class = "C2 (squared-exponential kernel)"

    def __init__(self, length_scale: float = 0.4, alpha: float = 1e-4) -> None:
        self.length_scale = length_scale
        self.alpha = alpha
        self.points: List[Point3D] = []
        self.alpha_weights: List[float] = []

    def _kernel(self, a: Point3D, b: Point3D) -> float:
        r = math.dist(a, b)
        return math.exp(-(r ** 2) / (2.0 * self.length_scale ** 2))

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        n = len(self.points)
        if n == 0:
            self.alpha_weights = []
            return
        kernel_matrix = [[self._kernel(self.points[i], self.points[j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            kernel_matrix[i][i] += self.alpha
        self.alpha_weights = solve_linear_system(kernel_matrix, list(values))

    def predict(self, point: Point3D) -> float:
        if not self.points:
            return 0.0
        k_star = [self._kernel(point, p) for p in self.points]
        return dot(k_star, self.alpha_weights)


def spherical_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h <= 0.0:
            return 0.0
        if h >= range_param:
            return sill
        ratio = h / range_param
        return sill * (1.5 * ratio - 0.5 * ratio ** 3)

    return _variogram


def exponential_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h <= 0.0:
            return 0.0
        ratio = h / range_param
        return sill * (1.0 - math.exp(-ratio))

    return _variogram


def gaussian_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h <= 0.0:
            return 0.0
        ratio = h / range_param
        return sill * (1.0 - math.exp(-(ratio ** 2)))

    return _variogram


def power_variogram(sill: float, power: float, scale: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h <= 0.0:
            return 0.0
        return sill * ((h / scale) ** power)

    return _variogram


def linear_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        ratio = min(max(h / range_param, 0.0), 1.0)
        return sill * ratio

    return _variogram


def cubic_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        t = min(max(h / range_param, 0.0), 1.0)
        value = (7 * (t ** 2) - 8 * (t ** 3) + 3 * (t ** 4)) / 2.0
        return sill * value if h < range_param else sill

    return _variogram


def rational_quadratic_variogram(
    sill: float, range_param: float, alpha: float = 1.0
) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        ratio = h / range_param
        return sill * ((ratio ** 2) / (alpha + (ratio ** 2)))

    return _variogram


def hole_effect_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h == 0:
            return 0.0
        if range_param <= 0:
            return sill
        scaled = h / range_param
        return sill * (1.0 - math.sin(scaled) / scaled)

    return _variogram


def matern_variogram(sill: float, range_param: float, nu: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        if h == 0:
            return 0.0
        r = h / range_param
        if nu == 1.5:
            coef = math.sqrt(3.0)
            return sill * (1.0 - (1.0 + coef * r) * math.exp(-coef * r))
        if nu == 2.5:
            coef = math.sqrt(5.0)
            return sill * (
                1.0
                - (
                    1.0
                    + coef * r
                    + 5.0 * (r ** 2) / 3.0
                )
                * math.exp(-coef * r)
            )
        return sill * (1.0 - math.exp(-r))

    return _variogram


def logarithmic_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill * math.log1p(h)
        return sill * math.log1p(h / range_param)

    return _variogram


def cauchy_variogram(sill: float, range_param: float) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        ratio = h / range_param
        return sill * (1.0 - 1.0 / (1.0 + ratio * ratio))

    return _variogram


def stable_variogram(
    sill: float, range_param: float, alpha: float = 1.2
) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if range_param <= 0:
            return sill
        ratio = (h / range_param) ** max(min(alpha, 2.0), 0.2)
        return min(sill, sill * ratio)

    return _variogram


def spline_variogram(sill: float, scale: float = 1.0) -> Callable[[float], float]:
    def _variogram(h: float) -> float:
        if h == 0:
            return 0.0
        return sill * (h ** 2) * math.log1p(h / max(scale, 1e-6))

    return _variogram


class BaseVariogramKriging(Interpolator):
    def __init__(
        self,
        *,
        name: str,
        variogram_fn: Callable[[float], float],
        smoothness: str = "C1 (variogram-based)",
        nugget: float = 1e-4,
    ) -> None:
        self.name = name
        self.smoothness_class = smoothness
        self.variogram_fn = variogram_fn
        self.nugget = nugget
        self.points: List[Point3D] = []
        self.values: List[float] = []

    def _distance(self, a: Point3D, b: Point3D) -> float:
        return math.dist(a, b)

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.points = list(points)
        self.values = list(values)

    def predict(self, point: Point3D) -> float:
        if not self.points:
            return 0.0
        n = len(self.points)
        matrix = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
        rhs = [0.0 for _ in range(n + 1)]
        for i in range(n):
            for j in range(n):
                h = self._distance(self.points[i], self.points[j])
                matrix[i][j] = self.variogram_fn(h)
            matrix[i][i] += self.nugget
            matrix[i][n] = 1.0
            matrix[n][i] = 1.0
        matrix[n][n] = 0.0
        for i in range(n):
            rhs[i] = self.variogram_fn(self._distance(point, self.points[i]))
        rhs[n] = 1.0
        solution = solve_linear_system(matrix, rhs)
        weights = solution[:n]
        return sum(w * v for w, v in zip(weights, self.values))


class OrdinaryKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.5, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Ordinary Kriging",
            smoothness="C1 (spherical variogram)",
            variogram_fn=spherical_variogram(sill, range_param),
            nugget=nugget,
        )


class ExponentialKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.45, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Exponential Kriging",
            smoothness="C\u221e (exponential variogram)",
            variogram_fn=exponential_variogram(sill, range_param),
            nugget=nugget,
        )


class GaussianKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.4, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Gaussian Kriging",
            smoothness="C\u221e (gaussian variogram)",
            variogram_fn=gaussian_variogram(sill, range_param),
            nugget=nugget,
        )


class PowerKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, power: float = 1.5, scale: float = 0.5, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Power Kriging",
            smoothness="C1 (fractional variogram)",
            variogram_fn=power_variogram(sill, power, scale),
            nugget=nugget,
        )


class AnisotropicKrigingInterpolator(BaseVariogramKriging):
    def __init__(
        self,
        ranges: Tuple[float, float, float] = (0.4, 0.25, 0.6),
        sill: float = 1.0,
        nugget: float = 1e-4,
    ) -> None:
        super().__init__(
            name="Anisotropic Kriging",
            smoothness="C\u221e (directional gaussian variogram)",
            variogram_fn=gaussian_variogram(sill, range_param=1.0),
            nugget=nugget,
        )
        self.axis_ranges = ranges

    def _distance(self, a: Point3D, b: Point3D) -> float:
        if not self.axis_ranges:
            return math.dist(a, b)
        scales = self.axis_ranges
        dx = (a[0] - b[0]) / scales[0]
        dy = (a[1] - b[1]) / scales[1]
        dz = (a[2] - b[2]) / scales[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


class UniversalKrigingInterpolator(Interpolator):
    name = "Universal Kriging"
    smoothness_class = "C1 (trend + variogram)"

    def __init__(self) -> None:
        self.base = OrdinaryKrigingInterpolator()

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        # Fit a simple linear trend first, then krige residuals.
        trend_model = LinearRegressionInterpolator()
        trend_model.fit(points, values)
        residuals = [v - trend_model.predict(p) for p, v in zip(points, values)]
        self.base.fit(points, residuals)
        self.trend_model = trend_model

    def predict(self, point: Point3D) -> float:
        return self.trend_model.predict(point) + self.base.predict(point)


class QuadraticTrendInterpolator(LinearRegressionInterpolator):
    def _design_row(self, point: Point3D) -> List[float]:
        x, y, z = point
        return [
            1.0,
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x * x,
            y * y,
            z * z,
        ]

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        design = [self._design_row(p) for p in points]
        if not design:
            self.coefficients = []
            return
        design_t = transpose(design)
        normal_matrix = matmul(design_t, design)
        # Add a small ridge term to avoid singular matrices on tiny datasets.
        for i in range(len(normal_matrix)):
            normal_matrix[i][i] += 1e-6
        normal_rhs = matvec(design_t, values)
        self.coefficients = solve_linear_system(normal_matrix, normal_rhs)


class QuadraticDriftKrigingInterpolator(Interpolator):
    name = "Universal Kriging (quadratic drift)"
    smoothness_class = "C1 (quadratic trend + residual kriging)"

    def __init__(self) -> None:
        self.trend_model = QuadraticTrendInterpolator()
        self.residual_model = GaussianKrigingInterpolator(range_param=0.35)

    def fit(self, points: Sequence[Point3D], values: Sequence[float]) -> None:
        self.trend_model.fit(points, values)
        residuals = [v - self.trend_model.predict(p) for p, v in zip(points, values)]
        self.residual_model.fit(points, residuals)

    def predict(self, point: Point3D) -> float:
        return self.trend_model.predict(point) + self.residual_model.predict(point)


class LinearKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.5, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Linear Kriging",
            smoothness="C0 (linear variogram)",
            variogram_fn=linear_variogram(sill, range_param),
            nugget=nugget,
        )


class CubicKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.6, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Cubic Kriging",
            smoothness="C2 (cubic variogram)",
            variogram_fn=cubic_variogram(sill, range_param),
            nugget=nugget,
        )


class RationalQuadraticKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.45, alpha: float = 1.0, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Rational Quadratic Kriging",
            smoothness="C\u221e (rational quadratic variogram)",
            variogram_fn=rational_quadratic_variogram(sill, range_param, alpha),
            nugget=nugget,
        )


class HoleEffectKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.35, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Hole-Effect Kriging",
            smoothness="C1 (oscillatory variogram)",
            variogram_fn=hole_effect_variogram(sill, range_param),
            nugget=nugget,
        )


class Matern32KrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.4, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Matérn ν=3/2 Kriging",
            smoothness="C2 (Matérn 3/2 variogram)",
            variogram_fn=matern_variogram(sill, range_param, nu=1.5),
            nugget=nugget,
        )


class Matern52KrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.35, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Matérn ν=5/2 Kriging",
            smoothness="C4 (Matérn 5/2 variogram)",
            variogram_fn=matern_variogram(sill, range_param, nu=2.5),
            nugget=nugget,
        )


class LogarithmicKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.5, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Logarithmic Kriging",
            smoothness="C0 (log variogram)",
            variogram_fn=logarithmic_variogram(sill, range_param),
            nugget=nugget,
        )


class CauchyKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.45, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Cauchy Kriging",
            smoothness="C\u221e (cauchy variogram)",
            variogram_fn=cauchy_variogram(sill, range_param),
            nugget=nugget,
        )


class StableKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, range_param: float = 0.4, alpha: float = 1.2, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Stable Kriging",
            smoothness="C1 (stable variogram)",
            variogram_fn=stable_variogram(sill, range_param, alpha),
            nugget=nugget,
        )


class SplineKrigingInterpolator(BaseVariogramKriging):
    def __init__(self, sill: float = 1.0, scale: float = 0.5, nugget: float = 1e-4) -> None:
        super().__init__(
            name="Spline Kriging",
            smoothness="C2 (thin-plate style variogram)",
            variogram_fn=spline_variogram(sill, scale),
            nugget=nugget,
        )


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def train_test_split(data: List[Tuple[Point3D, float]], test_ratio: float = 0.2) -> Tuple[List[Tuple[Point3D, float]], List[Tuple[Point3D, float]]]:
    n_test = max(1, int(len(data) * test_ratio))
    random.shuffle(data)
    return data[n_test:], data[:n_test]


def rmse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    errors = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    return math.sqrt(sum(errors) / len(errors)) if errors else 0.0


def create_grid(num: int, lower: float = 0.0, upper: float = 1.0) -> Tuple[List[float], float]:
    if num < 2:
        raise ValueError("Grid requires at least two points per axis.")
    step = (upper - lower) / (num - 1)
    coords = [lower + i * step for i in range(num)]
    return coords, step


def finite_difference_gradients(
    grid_values: Dict[Tuple[int, int, int], float],
    num: int,
    steps: Tuple[float, float, float],
) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    step_x, step_y, step_z = steps
    gradients: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            for k in range(1, num - 1):
                fx = (grid_values[(i + 1, j, k)] - grid_values[(i - 1, j, k)]) / (2 * step_x)
                fy = (grid_values[(i, j + 1, k)] - grid_values[(i, j - 1, k)]) / (2 * step_y)
                fz = (grid_values[(i, j, k + 1)] - grid_values[(i, j, k - 1)]) / (2 * step_z)
                gradients[(i, j, k)] = (fx, fy, fz)
    return gradients


def finite_difference_laplacian(
    grid_values: Dict[Tuple[int, int, int], float],
    num: int,
    steps: Tuple[float, float, float],
) -> Dict[Tuple[int, int, int], float]:
    step_x, step_y, step_z = steps
    laplacians: Dict[Tuple[int, int, int], float] = {}
    step_x2 = step_x * step_x
    step_y2 = step_y * step_y
    step_z2 = step_z * step_z
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            for k in range(1, num - 1):
                center = grid_values[(i, j, k)]
                laplacian = (
                    (grid_values[(i + 1, j, k)] - 2 * center + grid_values[(i - 1, j, k)]) / step_x2
                    + (grid_values[(i, j + 1, k)] - 2 * center + grid_values[(i, j - 1, k)]) / step_y2
                    + (grid_values[(i, j, k + 1)] - 2 * center + grid_values[(i, j, k - 1)]) / step_z2
                )
                laplacians[(i, j, k)] = laplacian
    return laplacians


def smoothness_metric_from_vectors(vectors: Dict[Tuple[int, int, int], Sequence[float]]) -> float:
    keys = sorted(vectors.keys())
    if not keys:
        return 0.0
    diffs: List[float] = []
    for key in keys:
        i, j, k = key
        neighbors = [
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ]
        current = vectors[key]
        for neighbor in neighbors:
            if neighbor in vectors:
                other = vectors[neighbor]
                diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, other)))
                diffs.append(diff)
    return statistics.mean(diffs) if diffs else 0.0


def smoothness_metric_from_scalars(values: Dict[Tuple[int, int, int], float]) -> float:
    keys = sorted(values.keys())
    diffs: List[float] = []
    for key in keys:
        i, j, k = key
        neighbors = [
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ]
        current = values[key]
        for neighbor in neighbors:
            if neighbor in values:
                diff = abs(current - values[neighbor])
                diffs.append(diff)
    return statistics.mean(diffs) if diffs else 0.0


@dataclass
class EvaluationResult:
    method: str
    smoothness_class: str
    rmse: float
    gradient_smoothness: float
    laplacian_smoothness: float
    grid_points: List[Point3D]
    grid_values: List[float]
    slice_axis: str
    slice_value: float
    slice_axis1_label: str
    slice_axis2_label: str
    slice_fixed_label: str
    slice_axis1_values: List[float]
    slice_axis2_values: List[float]
    slice_matrix: List[List[float]]


@dataclass
class TrainedMethod:
    interpolator: Interpolator
    method: str
    smoothness_class: str
    rmse: float
    gradient_smoothness: float
    laplacian_smoothness: float
    grid_points: List[Point3D]
    grid_values: List[float]


@dataclass
class LineSliceResult:
    method: str
    smoothness_class: str
    rmse: float
    gradient_smoothness: float
    laplacian_smoothness: float
    varying_axis_label: str
    varying_axis: str
    axis_values: List[float]
    predicted_values: List[float]
    fixed_axes: List[Tuple[str, float]]


@dataclass
class ComparisonSession:
    dataset: List[Tuple[Point3D, float]]
    dataset_source: str
    axis_bounds: List[Tuple[float, float]]
    grid_axes: List[List[float]]
    grid_size: int
    methods: List[TrainedMethod]
    skipped: List[Tuple[str, str]]


ACTIVE_SESSION: Optional[ComparisonSession] = None


def evaluate_interpolator(
    interpolator: Interpolator,
    train: List[Tuple[Point3D, float]],
    test: List[Tuple[Point3D, float]],
    grid_size: int = 6,
    *,
    axis_bounds: Sequence[Tuple[float, float]],
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
) -> EvaluationResult:
    if len(axis_bounds) != 3:
        raise ValueError("axis_bounds must contain bounds for X, Y, and Z.")

    train_points = [p for p, _ in train]
    train_values = [v for _, v in train]
    interpolator.fit(train_points, train_values)

    test_points = [p for p, _ in test]
    test_values = [v for _, v in test]
    predictions = [interpolator.predict(p) for p in test_points]
    error = rmse(predictions, test_values)

    grid_axes: List[List[float]] = []
    axis_steps: List[float] = []
    for lower, upper in axis_bounds:
        values, step = create_grid(grid_size, lower, upper)
        grid_axes.append(values)
        axis_steps.append(step if step > 0 else 1.0)

    xs, ys, zs = grid_axes
    step_x, step_y, step_z = axis_steps
    grid_values: Dict[Tuple[int, int, int], float] = {}
    grid_points_list: List[Point3D] = []
    grid_value_list: List[float] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                point = (x, y, z)
                value = interpolator.predict(point)
                grid_values[(i, j, k)] = value
                grid_points_list.append(point)
                grid_value_list.append(value)

    gradients = finite_difference_gradients(
        grid_values, grid_size, (step_x, step_y, step_z)
    )
    gradient_smoothness = smoothness_metric_from_vectors(gradients)

    laplacians = finite_difference_laplacian(
        grid_values, grid_size, (step_x, step_y, step_z)
    )
    laplacian_smoothness = smoothness_metric_from_scalars(laplacians)

    axis_lookup = {"x": 0, "y": 1, "z": 2}
    slice_axis_normalized = slice_axis.lower()
    drop_index = axis_lookup.get(slice_axis_normalized, 2)
    keep_indices = [index for index in range(3) if index != drop_index]
    axis1_index, axis2_index = keep_indices
    axis1_values = grid_axes[axis1_index][:]
    axis2_values = grid_axes[axis2_index][:]
    default_fixed = grid_axes[drop_index][len(grid_axes[drop_index]) // 2]
    fixed_value = slice_value if slice_value is not None else default_fixed

    slice_matrix: List[List[float]] = []
    for axis2 in axis2_values:
        row: List[float] = []
        for axis1 in axis1_values:
            coords = [0.0, 0.0, 0.0]
            coords[axis1_index] = axis1
            coords[axis2_index] = axis2
            coords[drop_index] = fixed_value
            row.append(interpolator.predict((coords[0], coords[1], coords[2])))
        slice_matrix.append(row)

    return EvaluationResult(
        method=getattr(interpolator, "name", interpolator.__class__.__name__),
        smoothness_class=getattr(interpolator, "smoothness_class", ""),
        rmse=error,
        gradient_smoothness=gradient_smoothness,
        laplacian_smoothness=laplacian_smoothness,
        grid_points=grid_points_list,
        grid_values=grid_value_list,
        slice_axis=AXIS_LABELS[drop_index],
        slice_value=fixed_value,
        slice_axis1_label=AXIS_LABELS[axis1_index],
        slice_axis2_label=AXIS_LABELS[axis2_index],
        slice_fixed_label=AXIS_LABELS[drop_index],
        slice_axis1_values=axis1_values,
        slice_axis2_values=axis2_values,
        slice_matrix=slice_matrix,
    )


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------


def create_interpolators() -> List[Interpolator]:
    return [
        NearestNeighborInterpolator(),
        KNNUniformInterpolator(k=4),
        InverseDistanceWeightingInterpolator(),
        LinearRegressionInterpolator(),
        RBFInterpolator("linear"),
        RBFInterpolator("cubic"),
        RBFInterpolator("quintic"),
        RBFInterpolator("gaussian", epsilon=1.5),
        RBFInterpolator("multiquadric", epsilon=1.0),
        RBFInterpolator("inverse_multiquadric", epsilon=1.0),
        ThinPlateSplineInterpolator(),
        GaussianProcessInterpolator(length_scale=0.35),
        OrdinaryKrigingInterpolator(),
        UniversalKrigingInterpolator(),
        ExponentialKrigingInterpolator(),
        GaussianKrigingInterpolator(),
        PowerKrigingInterpolator(),
        AnisotropicKrigingInterpolator(),
        QuadraticDriftKrigingInterpolator(),
        LinearKrigingInterpolator(),
        CubicKrigingInterpolator(),
        RationalQuadraticKrigingInterpolator(),
        HoleEffectKrigingInterpolator(),
        Matern32KrigingInterpolator(),
        Matern52KrigingInterpolator(),
        LogarithmicKrigingInterpolator(),
        CauchyKrigingInterpolator(),
        StableKrigingInterpolator(),
        SplineKrigingInterpolator(),
    ]


def compute_grid_axes(
    axis_bounds: Sequence[Tuple[float, float]],
    grid_size: int,
) -> Tuple[List[List[float]], List[float]]:
    grid_axes: List[List[float]] = []
    axis_steps: List[float] = []
    for lower, upper in axis_bounds:
        values, step = create_grid(grid_size, lower, upper)
        grid_axes.append(values)
        axis_steps.append(step if step > 0 else 1.0)
    return grid_axes, axis_steps


def compute_grid_predictions(
    interpolator: Interpolator,
    grid_axes: Sequence[Sequence[float]],
) -> Tuple[List[Point3D], List[float], Dict[Tuple[int, int, int], float]]:
    xs, ys, zs = grid_axes
    grid_values: Dict[Tuple[int, int, int], float] = {}
    grid_points_list: List[Point3D] = []
    grid_value_list: List[float] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                point = (x, y, z)
                value = interpolator.predict(point)
                grid_values[(i, j, k)] = value
                grid_points_list.append(point)
                grid_value_list.append(value)
    return grid_points_list, grid_value_list, grid_values


def train_single_interpolator(
    interpolator: Interpolator,
    train: List[Tuple[Point3D, float]],
    *,
    grid_axes: Sequence[Sequence[float]],
    axis_steps: Sequence[float],
    grid_size: int,
    full_dataset: Sequence[Tuple[Point3D, float]],
) -> TrainedMethod:
    train_points = [p for p, _ in train]
    train_values = [v for _, v in train]
    interpolator.fit(train_points, train_values)

    dataset_points = [p for p, _ in full_dataset]
    dataset_values = [v for _, v in full_dataset]
    predictions = [interpolator.predict(p) for p in dataset_points]
    error = rmse(predictions, dataset_values)

    grid_points, grid_value_list, grid_values_dict = compute_grid_predictions(
        interpolator, grid_axes
    )

    step_tuple = (
        axis_steps[0],
        axis_steps[1],
        axis_steps[2],
    )

    gradients = finite_difference_gradients(grid_values_dict, grid_size, step_tuple)
    gradient_smoothness = smoothness_metric_from_vectors(gradients)

    laplacians = finite_difference_laplacian(grid_values_dict, grid_size, step_tuple)
    laplacian_smoothness = smoothness_metric_from_scalars(laplacians)

    return TrainedMethod(
        interpolator=interpolator,
        method=getattr(interpolator, "name", interpolator.__class__.__name__),
        smoothness_class=getattr(interpolator, "smoothness_class", ""),
        rmse=error,
        gradient_smoothness=gradient_smoothness,
        laplacian_smoothness=laplacian_smoothness,
        grid_points=grid_points,
        grid_values=grid_value_list,
    )


def build_dataset_payload(
    dataset_list: Sequence[Tuple[Point3D, float]],
    dataset_source: str,
    axis_bounds: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    return {
        "points": [list(point) for point, _ in dataset_list],
        "values": [value for _, value in dataset_list],
        "source": dataset_source,
        "axis_bounds": list(axis_bounds),
    }


def serialize_results(results: List[EvaluationResult]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for item in results:
        payload.append(
            {
                "method": item.method,
                "smoothness_class": item.smoothness_class,
                "rmse": item.rmse,
                "gradient_smoothness": item.gradient_smoothness,
                "laplacian_smoothness": item.laplacian_smoothness,
                "grid_points": [list(point) for point in item.grid_points],
                "grid_values": item.grid_values[:],
                "slice": {
                    "axis": item.slice_axis.lower(),
                    "axis_label": item.slice_fixed_label,
                    "value": item.slice_value,
                    "axis1_label": item.slice_axis1_label,
                    "axis2_label": item.slice_axis2_label,
                    "axis1_values": item.slice_axis1_values,
                    "axis2_values": item.slice_axis2_values,
                    "matrix": item.slice_matrix,
                },
            }
        )
    return payload


def serialize_method_summaries(methods: Sequence[TrainedMethod]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for method in methods:
        payload.append(
            {
                "method": method.method,
                "smoothness_class": method.smoothness_class,
                "rmse": method.rmse,
                "gradient_smoothness": method.gradient_smoothness,
                "laplacian_smoothness": method.laplacian_smoothness,
            }
        )
    return payload


def serialize_line_results(results: List[LineSliceResult]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for item in results:
        payload.append(
            {
                "method": item.method,
                "smoothness_class": item.smoothness_class,
                "varying_axis": item.varying_axis.lower(),
                "varying_axis_label": item.varying_axis_label,
                "axis_values": item.axis_values[:],
                "predicted_values": item.predicted_values[:],
                "fixed_axes": [[label, value] for label, value in item.fixed_axes],
            }
        )
    return payload


def compute_prediction(
    session: ComparisonSession,
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
) -> Tuple[List[EvaluationResult], str, float]:
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    normalized_axis = (slice_axis or "z").lower()
    drop_index = axis_lookup.get(normalized_axis, 2)
    axis_values = session.grid_axes[drop_index][:]
    if axis_values:
        default_slice_value = axis_values[len(axis_values) // 2]
        min_bound = axis_values[0]
        max_bound = axis_values[-1]
    else:
        default_slice_value = 0.0
        min_bound = 0.0
        max_bound = 1.0

    if (
        slice_value is not None
        and isinstance(slice_value, (int, float))
        and math.isfinite(slice_value)
    ):
        candidate = float(slice_value)
        if min_bound <= max_bound:
            resolved_slice_value = max(min(candidate, max_bound), min_bound)
        else:
            resolved_slice_value = candidate
    else:
        resolved_slice_value = default_slice_value

    keep_indices = [index for index in range(3) if index != drop_index]
    axis1_index, axis2_index = keep_indices
    axis1_values = session.grid_axes[axis1_index][:]
    axis2_values = session.grid_axes[axis2_index][:]

    results: List[EvaluationResult] = []
    for method in session.methods:
        slice_matrix: List[List[float]] = []
        for axis2 in axis2_values:
            row: List[float] = []
            for axis1 in axis1_values:
                coords = [0.0, 0.0, 0.0]
                coords[axis1_index] = axis1
                coords[axis2_index] = axis2
                coords[drop_index] = resolved_slice_value
                row.append(method.interpolator.predict(tuple(coords)))
            slice_matrix.append(row)

        results.append(
            EvaluationResult(
                method=method.method,
                smoothness_class=method.smoothness_class,
                rmse=method.rmse,
                gradient_smoothness=method.gradient_smoothness,
                laplacian_smoothness=method.laplacian_smoothness,
                grid_points=method.grid_points,
                grid_values=method.grid_values,
                slice_axis=AXIS_LABELS[drop_index],
                slice_value=resolved_slice_value,
                slice_axis1_label=AXIS_LABELS[axis1_index],
                slice_axis2_label=AXIS_LABELS[axis2_index],
                slice_fixed_label=AXIS_LABELS[drop_index],
                slice_axis1_values=axis1_values,
                slice_axis2_values=axis2_values,
                slice_matrix=slice_matrix,
            )
        )

    return results, normalized_axis, resolved_slice_value


def _sanitize_line_resolution(candidate: Optional[float]) -> int:
    if candidate is None:
        return LINE_SLICE_RESOLUTION
    try:
        numeric = int(candidate)
    except (TypeError, ValueError):
        return LINE_SLICE_RESOLUTION
    return max(2, numeric)


def compute_line_predictions(
    session: ComparisonSession,
    varying_axis: str = "z",
    fixed_values: Optional[Dict[str, float]] = None,
    line_resolution: Optional[int] = None,
) -> Tuple[List[LineSliceResult], str, Dict[str, float], int]:
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    normalized_axis = (varying_axis or "z").lower()
    varying_index = axis_lookup.get(normalized_axis, 2)

    lower, upper = session.axis_bounds[varying_index]
    sanitized_resolution = _sanitize_line_resolution(line_resolution)
    effective_resolution = max(session.grid_size, sanitized_resolution)
    axis_values, _ = create_grid(effective_resolution, lower, upper)

    normalized_fixed_input: Dict[str, float] = {}
    if isinstance(fixed_values, dict):
        for key, value in fixed_values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                normalized_fixed_input[key.lower()] = float(value)

    resolved_fixed: Dict[str, float] = {}
    coords_template = [0.0, 0.0, 0.0]
    for axis_index, axis_label in enumerate(AXIS_LABELS):
        label_lower = axis_label.lower()
        if axis_index == varying_index:
            continue
        candidate = normalized_fixed_input.get(label_lower)
        lower, upper = session.axis_bounds[axis_index]
        min_bound = min(lower, upper)
        max_bound = max(lower, upper)
        if candidate is None:
            axis_entries = session.grid_axes[axis_index]
            if axis_entries:
                candidate = axis_entries[len(axis_entries) // 2]
            else:
                candidate = (lower + upper) / 2.0
        else:
            candidate = max(min(candidate, max_bound), min_bound)
        coords_template[axis_index] = candidate
        resolved_fixed[label_lower] = candidate

    results: List[LineSliceResult] = []
    for method in session.methods:
        predicted_values: List[float] = []
        for axis_value in axis_values:
            coords = coords_template[:]
            coords[varying_index] = axis_value
            predicted_values.append(method.interpolator.predict(tuple(coords)))

        fixed_axes_info = [
            (label, resolved_fixed[label.lower()])
            for label in AXIS_LABELS
            if label.lower() in resolved_fixed
        ]

        results.append(
            LineSliceResult(
                method=method.method,
                smoothness_class=method.smoothness_class,
                rmse=method.rmse,
                gradient_smoothness=method.gradient_smoothness,
                laplacian_smoothness=method.laplacian_smoothness,
                varying_axis_label=AXIS_LABELS[varying_index],
                varying_axis=AXIS_LABELS[varying_index],
                axis_values=axis_values[:],
                predicted_values=predicted_values,
                fixed_axes=fixed_axes_info,
            )
        )

    return results, normalized_axis, resolved_fixed, effective_resolution


def fit_session(
    dataset: Optional[Sequence[Tuple[Point3D, float]]] = None,
    *,
    num_points: int = 10,
    seed: int = 123,
    test_ratio: float = 0.2,
    grid_size: int = 6,
) -> ComparisonSession:
    if dataset is None:
        dataset_list = generate_dataset(num_points, seed=seed)
        dataset_source = "synthetic"
    else:
        dataset_list = normalize_dataset(dataset)
        dataset_source = "custom"

    axis_bounds = compute_axis_bounds([point for point, _ in dataset_list])
    grid_axes, axis_steps = compute_grid_axes(axis_bounds, grid_size)
    train = dataset_list[:]  # Fit on all points so RMSE reflects original samples

    methods: List[TrainedMethod] = []
    skipped: List[Tuple[str, str]] = []
    for interpolator in create_interpolators():
        try:
            trained = train_single_interpolator(
                interpolator,
                train[:],
                grid_axes=grid_axes,
                axis_steps=axis_steps,
                grid_size=grid_size,
                full_dataset=dataset_list,
            )
            methods.append(trained)
        except ValueError as exc:
            skipped.append((interpolator.name, str(exc)))

    session = ComparisonSession(
        dataset=dataset_list,
        dataset_source=dataset_source,
        axis_bounds=list(axis_bounds),
        grid_axes=grid_axes,
        grid_size=grid_size,
        methods=methods,
        skipped=skipped,
    )

    global ACTIVE_SESSION
    ACTIVE_SESSION = session
    return session


def predict_session(
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
    *,
    session: Optional[ComparisonSession] = None,
) -> Dict[str, object]:
    active_session = session or ACTIVE_SESSION
    if active_session is None:
        raise RuntimeError("No active session available. Call fit_session() first.")

    results, normalized_axis, resolved_value = compute_prediction(
        active_session, slice_axis, slice_value
    )

    dataset_payload = build_dataset_payload(
        active_session.dataset,
        active_session.dataset_source,
        active_session.axis_bounds,
    )
    svg_content = render_bar_chart_svg_string(results) if results else ""

    return {
        "results": serialize_results(results),
        "svg": svg_content,
        "dataset": dataset_payload,
        "skipped": active_session.skipped,
        "dataset_source": active_session.dataset_source,
        "slice_axis": normalized_axis,
        "slice_value": resolved_value,
    }


def predict_line_session(
    varying_axis: str = "z",
    fixed_values: Optional[Dict[str, float]] = None,
    line_resolution: Optional[int] = None,
    *,
    session: Optional[ComparisonSession] = None,
) -> Dict[str, object]:
    active_session = session or ACTIVE_SESSION
    if active_session is None:
        raise RuntimeError("No active session available. Call fit_session() first.")

    (
        line_results,
        normalized_axis,
        resolved_fixed,
        resolved_resolution,
    ) = compute_line_predictions(
        active_session,
        varying_axis,
        fixed_values,
        line_resolution=line_resolution,
    )

    dataset_payload = build_dataset_payload(
        active_session.dataset,
        active_session.dataset_source,
        active_session.axis_bounds,
    )

    return {
        "line_results": serialize_line_results(line_results),
        "dataset": dataset_payload,
        "dataset_source": active_session.dataset_source,
        "skipped": active_session.skipped,
        "line_axis": normalized_axis,
        "fixed_axes": resolved_fixed,
        "summaries": serialize_method_summaries(active_session.methods),
        "line_resolution": resolved_resolution,
    }


# ---------------------------------------------------------------------------
# Visualization (SVG chart)
# ---------------------------------------------------------------------------


def render_bar_chart_svg_string(results: List[EvaluationResult]) -> str:
    if not results:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='200'><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'>No results available</text></svg>"

    width = 1000
    height = 600
    margin = 80
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    bar_group_height = chart_height / 3.0

    # Prepare scaling
    rmse_max = max(result.rmse for result in results) or 1.0
    grad_max = max(result.gradient_smoothness for result in results) or 1.0
    lap_max = max(result.laplacian_smoothness for result in results) or 1.0

    def bar_height(value: float, max_value: float) -> float:
        return (value / max_value) * (bar_group_height - 40)

    labels = [result.method for result in results]
    num_methods = len(labels)
    bar_width = chart_width / (num_methods * 1.2)

    def bar_x(index: int) -> float:
        return margin + index * bar_width * 1.2

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial;font-size:12px;} .title{font-size:18px;font-weight:bold;}</style>",
        f"<text class='title' x='{width/2}' y='30' text-anchor='middle'>3D Interpolation Comparison</text>",
    ]

    metrics = [
        ("RMSE (lower is better)", [result.rmse for result in results], rmse_max),
        ("Gradient smoothness (lower is smoother)", [result.gradient_smoothness for result in results], grad_max),
        ("Laplacian smoothness (lower is smoother)", [result.laplacian_smoothness for result in results], lap_max),
    ]

    for group_idx, (title, values, max_value) in enumerate(metrics):
        y_offset = margin + group_idx * bar_group_height
        parts.append(f"<text x='{margin}' y='{y_offset - 20}'>{title}</text>")
        for i, value in enumerate(values):
            h = bar_height(value, max_value)
            x = bar_x(i)
            y = y_offset + (bar_group_height - 40) - h
            color = f"hsl({(i / max(1, num_methods)) * 360:.0f},60%,60%)"
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{h:.1f}' fill='{color}' />"
            )
            parts.append(
                f"<text x='{x + bar_width/2:.1f}' y='{y - 4:.1f}' text-anchor='middle'>{value:.3f}</text>"
            )
        for i, label in enumerate(labels):
            x = bar_x(i) + bar_width / 2
            parts.append(
                f"<text x='{x:.1f}' y='{y_offset + bar_group_height - 15:.1f}' text-anchor='middle' transform='rotate(45 {x:.1f} {y_offset + bar_group_height - 15:.1f})'>{label}</text>"
            )

    parts.append("</svg>")

    return "\n".join(parts)


def render_bar_chart_svg(results: List[EvaluationResult], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    svg_content = render_bar_chart_svg_string(results)
    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(svg_content)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def run_comparison(
    num_points: int = 10,
    seed: int = 123,
    test_ratio: float = 0.2,
    grid_size: int = 6,
    save_artifacts: bool = True,
    dataset: Optional[Sequence[Tuple[Point3D, float]]] = None,
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
) -> Dict[str, object]:
    session = fit_session(
        dataset=dataset,
        num_points=num_points,
        seed=seed,
        test_ratio=test_ratio,
        grid_size=grid_size,
    )
    if dataset is None:
        csv_path = os.path.join("data", "dummy_3d_points.csv")
        if save_artifacts:
            write_csv(csv_path, session.dataset)
    else:
        csv_path = None

    results, normalized_axis, resolved_slice_value = compute_prediction(
        session, slice_axis, slice_value
    )
    results_path = os.path.join("output", "results_summary.csv")
    svg_path = os.path.join("output", "interpolation_comparison.svg")

    if save_artifacts and results:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Method", "Smoothness", "RMSE", "GradientSmoothness", "LaplacianSmoothness"])
            for result in results:
                writer.writerow(
                    [
                        result.method,
                        result.smoothness_class,
                        f"{result.rmse:.6f}",
                        f"{result.gradient_smoothness:.6f}",
                        f"{result.laplacian_smoothness:.6f}",
                    ]
                )

        render_bar_chart_svg(results, svg_path)

    svg_content = render_bar_chart_svg_string(results) if results else ""
    results_payload = serialize_results(results)
    dataset_payload = build_dataset_payload(
        session.dataset, session.dataset_source, session.axis_bounds
    )

    results_csv_actual = results_path if save_artifacts and results else None
    svg_path_actual = svg_path if save_artifacts and results else None

    return {
        "dataset_csv_path": csv_path if save_artifacts else None,
        "results_csv_path": results_csv_actual,
        "svg_path": svg_path_actual,
        "results": results_payload,
        "svg": svg_content,
        "dataset": dataset_payload,
        "skipped": session.skipped,
        "dataset_source": session.dataset_source,
        "slice_axis": normalized_axis,
        "slice_value": resolved_slice_value,
    }


def main() -> None:
    summary = run_comparison()

    if summary["dataset_source"] == "synthetic":
        print("Synthetic dataset written to:", summary["dataset_csv_path"])
    else:
        print(
            "Custom dataset supplied with",
            len(summary["dataset"]["points"]),
            "rows.",
        )
    print("Evaluation summary written to:", summary["results_csv_path"])
    print("SVG chart written to:", summary["svg_path"])
    print()
    print("Summary:")
    for result in summary["results"]:
        print(
            f"- {result['method']} | {result['smoothness_class']}: "
            f"RMSE={result['rmse']:.4f}, "
            f"GradSmooth={result['gradient_smoothness']:.4f}, "
            f"LapSmooth={result['laplacian_smoothness']:.4f}"
        )
    if summary["skipped"]:
        print()
        print("Skipped methods:")
        for name, message in summary["skipped"]:
            print(f"- {name}: {message}")


if __name__ == "__main__":
    main()
