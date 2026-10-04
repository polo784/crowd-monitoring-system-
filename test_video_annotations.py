"""Verify count matching and prevent draft annotations becoming accuracy scores."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from compare_video_annotations import compare


class AnnotationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.annotations = self.root / "annotations.json"
        self.measurements = self.root / "measurements.csv"
        self.document = {"frames": []}
        for index in (0, 165):
            name = f"frame_{index}.jpg"
            cv2.imwrite(str(self.root / name), np.zeros((48, 64, 3), dtype=np.uint8))
            self.document["frames"].append({
                "frame_index": index, "image": name, "width": 64, "height": 48,
                "reviewed": False, "complete": False, "unresolved_regions": [],
                "points": [{"id": 1, "x": 10, "y": 10, "uncertain": False},
                           {"id": 2, "x": 30, "y": 30, "uncertain": False}],
            })
        with self.measurements.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame_index", "timestamp_seconds", "predicted_count"])
            writer.writeheader()
            # Reversed row order catches comparisons that match by position.
            writer.writerows([{"frame_index": 165, "timestamp_seconds": 5.5, "predicted_count": 4.0},
                              {"frame_index": 0, "timestamp_seconds": 0, "predicted_count": 3.5}])

    def tearDown(self):
        self.temporary.cleanup()

    def run_comparison(self, name="out"):
        self.annotations.write_text(json.dumps(self.document))
        return compare(self.annotations, self.measurements, self.root / name)

    def test_drafts_have_no_accuracy_metrics_and_match_frame_indexes(self):
        result = self.run_comparison()
        self.assertIsNone(result["mae"])
        self.assertIsNone(result["rmse"])
        self.assertEqual([r["predicted_count"] for r in result["sampled_frames"]], [3.5, 4.0])
        self.assertEqual(result["mean_absolute_difference_from_draft"], 1.75)

    def test_reviewed_complete_points_produce_metrics(self):
        for frame in self.document["frames"]:
            frame.update(reviewed=True, complete=True)
        result = self.run_comparison()
        self.assertEqual(result["mae"], 1.75)
        self.assertAlmostEqual(result["rmse"], math.sqrt((1.5**2 + 2.0**2) / 2))

    def test_uncertain_or_unresolved_annotations_block_metrics(self):
        for frame in self.document["frames"]:
            frame.update(reviewed=True, complete=True)
        self.document["frames"][0]["points"][0]["uncertain"] = True
        self.assertIsNone(self.run_comparison("uncertain")["mae"])
        self.document["frames"][0]["points"][0]["uncertain"] = False
        self.document["frames"][0]["unresolved_regions"] = [{"label": "unreviewed"}]
        self.assertIsNone(self.run_comparison("unresolved")["mae"])

    def test_missing_prediction_is_rejected(self):
        self.document["frames"][0]["frame_index"] = 99
        with self.assertRaisesRegex(ValueError, "No saved model prediction"):
            self.run_comparison()

    def test_out_of_bounds_point_is_rejected(self):
        self.document["frames"][0]["points"][0]["x"] = 64
        with self.assertRaisesRegex(ValueError, "Invalid head point"):
            self.run_comparison()


if __name__ == "__main__":
    unittest.main()
