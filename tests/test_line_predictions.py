"""Regression tests for line-slice predictions."""

from __future__ import annotations

import unittest
import asyncio

from docs.compare_interpolation import (
    create_interpolators,
    fit_session,
    predict_line_session,
    predict_line_session_async,
    predict_session,
)


class LinePredictionSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = []
        for index in range(12):
            x = 10.0 + index * 0.75
            y = -5.0 + index * 0.4
            z = 2.0 + index * 0.25
            value = 0.5 * x - 0.25 * y + 0.1 * z
            self.dataset.append(((x, y, z), value))
        xs, ys, zs = zip(*[point for point, _ in self.dataset])
        self.bounds = {
            "x": (min(xs), max(xs)),
            "y": (min(ys), max(ys)),
            "z": (min(zs), max(zs)),
        }

    def test_line_predictions_use_custom_bounds(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=4, test_ratio=0.25)
        result = predict_line_session(
            session=session,
            varying_axis="x",
            fixed_values={"y": 999.0, "z": -999.0},
        )

        line_results = result["line_results"]
        self.assertTrue(line_results, "line results should not be empty")
        axis_values = line_results[0]["axis_values"]
        self.assertAlmostEqual(axis_values[0], self.bounds["x"][0])
        self.assertAlmostEqual(axis_values[-1], self.bounds["x"][1])

        dataset_payload = result["dataset"]
        self.assertEqual(dataset_payload["source"], "custom")
        self.assertEqual(len(dataset_payload["points"]), len(self.dataset))
        first_point = dataset_payload["points"][0]
        self.assertAlmostEqual(first_point[0], self.dataset[0][0][0])

    def test_fixed_axes_are_clamped_to_dataset_range(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=5, test_ratio=0.25)
        result = predict_line_session(
            session=session,
            varying_axis="z",
            fixed_values={"x": -100.0, "y": 100.0},
        )

        fixed_axes = result["fixed_axes"]
        self.assertGreaterEqual(fixed_axes["x"], self.bounds["x"][0])
        self.assertLessEqual(fixed_axes["x"], self.bounds["x"][1])
        self.assertGreaterEqual(fixed_axes["y"], self.bounds["y"][0])
        self.assertLessEqual(fixed_axes["y"], self.bounds["y"][1])

    def test_line_resolution_override(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=5, test_ratio=0.25)
        requested_resolution = 150
        result = predict_line_session(
            session=session,
            varying_axis="y",
            line_resolution=requested_resolution,
        )

        line_results = result["line_results"]
        self.assertTrue(line_results, "line results should not be empty")
        axis_values = line_results[0]["axis_values"]
        expected_length = max(5, requested_resolution)
        self.assertEqual(len(axis_values), expected_length)
        self.assertEqual(result["line_resolution"], expected_length)

    def test_predict_session_filters_to_selected_algorithms(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=4, test_ratio=0.25)

        result = predict_session(
            session=session,
            slice_axis="z",
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["method"], "Inverse Distance Weighting")

    def test_predict_line_session_filters_to_selected_algorithms(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=4, test_ratio=0.25)

        result = predict_line_session(
            session=session,
            varying_axis="x",
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        self.assertEqual(len(result["line_results"]), 1)
        self.assertEqual(result["line_results"][0]["method"], "Inverse Distance Weighting")
        self.assertEqual(len(result["summaries"]), 1)
        self.assertEqual(result["summaries"][0]["method"], "Inverse Distance Weighting")

    def test_predict_line_session_async_filters_summaries_to_selected_algorithms(self) -> None:
        session = fit_session(dataset=self.dataset, grid_size=4, test_ratio=0.25)

        result = asyncio.run(
            predict_line_session_async(
                session=session,
                varying_axis="x",
                algorithm_configs=[
                    {
                        "id": "IDW",
                        "enabled": True,
                        "params": {"power": 2.0},
                    }
                ],
            )
        )

        self.assertEqual(len(result["line_results"]), 1)
        self.assertEqual(result["line_results"][0]["method"], "Inverse Distance Weighting")
        self.assertEqual(len(result["summaries"]), 1)
        self.assertEqual(result["summaries"][0]["method"], "Inverse Distance Weighting")

    def test_fit_session_uses_only_explicitly_selected_algorithms(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                },
                {
                    "id": "NearestNeighbor",
                    "enabled": False,
                    "params": {},
                },
                {
                    "id": "KNNUniform",
                    "enabled": False,
                    "params": {"k": 4},
                },
            ],
        )

        self.assertEqual(len(session.methods), 1)
        self.assertEqual(session.methods[0].method, "Inverse Distance Weighting")

    def test_create_interpolators_uses_only_explicitly_selected_algorithms(self) -> None:
        interpolators = create_interpolators(
            [
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 3.0},
                }
            ]
        )

        self.assertEqual(len(interpolators), 1)
        self.assertEqual(interpolators[0].name, "Inverse Distance Weighting")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
