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


class OrdinaryKrigingInterpolator(Interpolator):
    name = "Ordinary Kriging"
    smoothness_class = "C1 (variogram-based)"

    def __init__(self, sill: float = 1.0, range_param: float = 0.5, nugget: float = 1e-4) -> None:
        self.sill = sill
        self.range_param = range_param
        self.nugget = nugget
        self.points: List[Point3D] = []
        self.values: List[float] = []

    def _variogram(self, h: float) -> float:
        # Spherical variogram model
        if h >= self.range_param:
            return self.sill
        ratio = h / self.range_param
        return self.sill * (1.5 * ratio - 0.5 * ratio ** 3)

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
                h = math.dist(self.points[i], self.points[j])
                matrix[i][j] = self._variogram(h)
            matrix[i][i] += self.nugget
            matrix[i][n] = 1.0
            matrix[n][i] = 1.0
        matrix[n][n] = 0.0
        for i in range(n):
            rhs[i] = self._variogram(math.dist(point, self.points[i]))
        rhs[n] = 1.0
        solution = solve_linear_system(matrix, rhs)
        weights = solution[:n]
        return sum(w * v for w, v in zip(weights, self.values))


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


def finite_difference_gradients(grid_values: Dict[Tuple[int, int, int], float], num: int, step: float) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    gradients: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            for k in range(1, num - 1):
                fx = (grid_values[(i + 1, j, k)] - grid_values[(i - 1, j, k)]) / (2 * step)
                fy = (grid_values[(i, j + 1, k)] - grid_values[(i, j - 1, k)]) / (2 * step)
                fz = (grid_values[(i, j, k + 1)] - grid_values[(i, j, k - 1)]) / (2 * step)
                gradients[(i, j, k)] = (fx, fy, fz)
    return gradients


def finite_difference_laplacian(grid_values: Dict[Tuple[int, int, int], float], num: int, step: float) -> Dict[Tuple[int, int, int], float]:
    laplacians: Dict[Tuple[int, int, int], float] = {}
    step2 = step * step
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            for k in range(1, num - 1):
                center = grid_values[(i, j, k)]
                laplacian = (
                    grid_values[(i + 1, j, k)]
                    + grid_values[(i - 1, j, k)]
                    + grid_values[(i, j + 1, k)]
                    + grid_values[(i, j - 1, k)]
                    + grid_values[(i, j, k + 1)]
                    + grid_values[(i, j, k - 1)]
                    - 6 * center
                ) / step2
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


def evaluate_interpolator(
    interpolator: Interpolator,
    train: List[Tuple[Point3D, float]],
    test: List[Tuple[Point3D, float]],
    grid_size: int = 6,
) -> EvaluationResult:
    train_points = [p for p, _ in train]
    train_values = [v for _, v in train]
    interpolator.fit(train_points, train_values)

    test_points = [p for p, _ in test]
    test_values = [v for _, v in test]
    predictions = [interpolator.predict(p) for p in test_points]
    error = rmse(predictions, test_values)

    xs, step = create_grid(grid_size)
    grid_values: Dict[Tuple[int, int, int], float] = {}
    grid_points_list: List[Point3D] = []
    grid_value_list: List[float] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            for k, z in enumerate(xs):
                point = (x, y, z)
                value = interpolator.predict(point)
                grid_values[(i, j, k)] = value
                grid_points_list.append(point)
                grid_value_list.append(value)

    gradients = finite_difference_gradients(grid_values, grid_size, step)
    gradient_smoothness = smoothness_metric_from_vectors(gradients)

    laplacians = finite_difference_laplacian(grid_values, grid_size, step)
    laplacian_smoothness = smoothness_metric_from_scalars(laplacians)

    return EvaluationResult(
        method=getattr(interpolator, "name", interpolator.__class__.__name__),
        smoothness_class=getattr(interpolator, "smoothness_class", ""),
        rmse=error,
        gradient_smoothness=gradient_smoothness,
        laplacian_smoothness=laplacian_smoothness,
        grid_points=grid_points_list,
        grid_values=grid_value_list,
    )


# ---------------------------------------------------------------------------
# Visualization (SVG chart)
# ---------------------------------------------------------------------------


def render_bar_chart_svg_string(results: List[EvaluationResult]) -> str:
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
) -> Dict[str, object]:
    if dataset is None:
        dataset_list = generate_dataset(num_points, seed=seed)
        dataset_source = "synthetic"
        csv_path = os.path.join("data", "dummy_3d_points.csv")
        if save_artifacts:
            write_csv(csv_path, dataset_list)
    else:
        dataset_list = normalize_dataset(dataset)
        dataset_source = "custom"
        csv_path = None

    train, test = train_test_split(dataset_list[:], test_ratio=test_ratio)

    interpolators: List[Interpolator] = [
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
        RBFInterpolator("thin_plate_spline"),
        GaussianProcessInterpolator(length_scale=0.35),
        OrdinaryKrigingInterpolator(),
        UniversalKrigingInterpolator(),
    ]

    results: List[EvaluationResult] = []
    skipped: List[Tuple[str, str]] = []
    for interpolator in interpolators:
        try:
            result = evaluate_interpolator(interpolator, train[:], test[:], grid_size=grid_size)
            results.append(result)
        except ValueError as exc:
            skipped.append((interpolator.name, str(exc)))

    results_path = os.path.join("output", "results_summary.csv")
    svg_path = os.path.join("output", "interpolation_comparison.svg")

    if save_artifacts:
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

    svg_content = render_bar_chart_svg_string(results)
    results_payload = [
        {
            "method": item.method,
            "smoothness_class": item.smoothness_class,
            "rmse": item.rmse,
            "gradient_smoothness": item.gradient_smoothness,
            "laplacian_smoothness": item.laplacian_smoothness,
            "grid_points": [list(point) for point in item.grid_points],
            "grid_values": item.grid_values,
        }
        for item in results
    ]

    dataset_payload = {
        "points": [list(point) for point, _ in dataset_list],
        "values": [value for _, value in dataset_list],
        "source": dataset_source,
    }

    return {
        "dataset_csv_path": csv_path if save_artifacts else None,
        "results_csv_path": results_path if save_artifacts else None,
        "svg_path": svg_path if save_artifacts else None,
        "results": results_payload,
        "svg": svg_content,
        "dataset": dataset_payload,
        "skipped": skipped,
        "dataset_source": dataset_source,
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
