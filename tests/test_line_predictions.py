"""Regression tests for line-slice predictions."""

from __future__ import annotations

import unittest
import asyncio

from docs.compare_interpolation import (
    create_interpolators,
    export_session,
    fit_session,
    import_session,
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

    def test_normalized_line_predictions_return_normalized_display_coordinates(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        result = predict_line_session(
            session=session,
            varying_axis="x",
            fixed_values={"y": 0.0, "z": 0.0},
        )

        expected_x_min = (self.bounds["x"][0] - session.norm_means[0]) / session.norm_stds[0]
        expected_x_max = (self.bounds["x"][1] - session.norm_means[0]) / session.norm_stds[0]
        expected_first_point = [
            (self.dataset[0][0][axis] - session.norm_means[axis]) / session.norm_stds[axis]
            for axis in range(3)
        ]

        self.assertEqual(len(result["line_results"]), 1)
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][0], expected_x_min)
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][-1], expected_x_max)
        self.assertAlmostEqual(result["fixed_axes"]["y"], 0.0)
        self.assertAlmostEqual(result["fixed_axes"]["z"], 0.0)
        self.assertAlmostEqual(result["dataset"]["axis_bounds"][0][0], expected_x_min)
        self.assertAlmostEqual(result["dataset"]["axis_bounds"][0][1], expected_x_max)
        for axis in range(3):
            self.assertAlmostEqual(result["dataset"]["points"][0][axis], expected_first_point[axis])

    def test_denormalized_line_predictions_return_original_display_coordinates(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        expected_original_y = self.bounds["y"][0] + (self.bounds["y"][1] - self.bounds["y"][0]) / 2.0
        expected_original_z = self.bounds["z"][0] + (self.bounds["z"][1] - self.bounds["z"][0]) / 2.0

        result = predict_line_session(
            session=session,
            varying_axis="x",
            fixed_values={"y": expected_original_y, "z": expected_original_z},
            denormalize_after_predict=True,
        )

        self.assertEqual(len(result["line_results"]), 1)
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][0], self.bounds["x"][0])
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][-1], self.bounds["x"][1])
        self.assertAlmostEqual(result["fixed_axes"]["y"], expected_original_y)
        self.assertAlmostEqual(result["fixed_axes"]["z"], expected_original_z)
        self.assertAlmostEqual(result["dataset"]["axis_bounds"][0][0], self.bounds["x"][0])
        self.assertAlmostEqual(result["dataset"]["axis_bounds"][0][1], self.bounds["x"][1])
        self.assertAlmostEqual(result["dataset"]["points"][0][0], self.dataset[0][0][0])

    def test_normalized_plane_predictions_use_normalized_slice_coordinates(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        result = predict_session(
            session=session,
            slice_axis="z",
            slice_value=0.0,
        )

        slice_payload = result["results"][0]["slice"]
        expected_x_min = (self.bounds["x"][0] - session.norm_means[0]) / session.norm_stds[0]
        expected_x_max = (self.bounds["x"][1] - session.norm_means[0]) / session.norm_stds[0]
        expected_y_min = (self.bounds["y"][0] - session.norm_means[1]) / session.norm_stds[1]
        expected_y_max = (self.bounds["y"][1] - session.norm_means[1]) / session.norm_stds[1]

        self.assertEqual(len(result["results"]), 1)
        self.assertAlmostEqual(result["slice_value"], 0.0)
        self.assertAlmostEqual(slice_payload["value"], 0.0)
        self.assertAlmostEqual(slice_payload["axis1_values"][0], expected_x_min)
        self.assertAlmostEqual(slice_payload["axis1_values"][-1], expected_x_max)
        self.assertAlmostEqual(slice_payload["axis2_values"][0], expected_y_min)
        self.assertAlmostEqual(slice_payload["axis2_values"][-1], expected_y_max)

    def test_denormalized_plane_predictions_use_original_slice_coordinates(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        original_slice_value = self.bounds["z"][0] + (self.bounds["z"][1] - self.bounds["z"][0]) / 2.0
        result = predict_session(
            session=session,
            slice_axis="z",
            slice_value=original_slice_value,
            denormalize_after_predict=True,
        )

        slice_payload = result["results"][0]["slice"]
        self.assertEqual(len(result["results"]), 1)
        self.assertAlmostEqual(result["slice_value"], original_slice_value)
        self.assertAlmostEqual(slice_payload["value"], original_slice_value)
        self.assertAlmostEqual(slice_payload["axis1_values"][0], self.bounds["x"][0])
        self.assertAlmostEqual(slice_payload["axis1_values"][-1], self.bounds["x"][1])
        self.assertAlmostEqual(slice_payload["axis2_values"][0], self.bounds["y"][0])
        self.assertAlmostEqual(slice_payload["axis2_values"][-1], self.bounds["y"][1])

    def test_exported_and_imported_session_preserves_selected_algorithm_ids(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        exported = export_session(session)
        restored = import_session(exported)
        result = predict_session(session=restored, slice_axis="z")

        self.assertEqual(len(restored.methods), 1)
        self.assertEqual(restored.methods[0].algorithm_id, "IDW")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["method"], "Inverse Distance Weighting")

    def test_imported_normalized_session_uses_normalized_display_coordinates(self) -> None:
        session = fit_session(
            dataset=self.dataset,
            grid_size=4,
            test_ratio=0.25,
            normalize=True,
            algorithm_configs=[
                {
                    "id": "IDW",
                    "enabled": True,
                    "params": {"power": 2.0},
                }
            ],
        )

        restored = import_session(export_session(session))
        result = predict_line_session(
            session=restored,
            varying_axis="x",
            fixed_values={"y": 0.0, "z": 0.0},
        )
        expected_x_min = (self.bounds["x"][0] - session.norm_means[0]) / session.norm_stds[0]
        expected_x_max = (self.bounds["x"][1] - session.norm_means[0]) / session.norm_stds[0]

        self.assertAlmostEqual(result["fixed_axes"]["y"], 0.0)
        self.assertAlmostEqual(result["fixed_axes"]["z"], 0.0)
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][0], expected_x_min)
        self.assertAlmostEqual(result["line_results"][0]["axis_values"][-1], expected_x_max)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
