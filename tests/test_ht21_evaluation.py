"""Contract checks for CroHD counting rules and evaluation safeguards."""

import unittest

import numpy as np

from evaluate_ht21 import count_metrics, human_masks, validate_ground_truth


class HT21EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([
            [1, 1, 0, 0, 10, 10, 1, 1, 1],  # visible walking person
            [1, 2, 0, 0, 10, 10, 1, 2, 1],  # visible stationary person
            [1, 3, 0, 0, 10, 10, 1, 3, 1],  # ignored non-person
            [2, 1, 0, 0, 10, 10, 1, 1, 0],  # fully occluded tracked person
            [2, 4, 0, 0, 10, 10, 1, 4, 1],  # person on vehicle
            [2, 5, 0, 0, 10, 10, 0, 1, 1],  # invalid annotation flag
        ], dtype=float)

    def test_people_classes_visibility_and_confidence(self):
        validate_ground_truth(self.rows, 2)
        visible, people = human_masks(self.rows)
        np.testing.assert_array_equal(visible, [True, True, False, False, True, False])
        np.testing.assert_array_equal(people, [True, True, False, True, True, False])

    def test_counting_metrics_keep_signed_bias(self):
        metrics = count_metrics([8, 14], [10, 10])
        self.assertEqual(metrics["mae"], 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(10))
        self.assertEqual(metrics["mean_signed_error"], 1)

    def test_duplicate_frame_person_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_ground_truth(np.concatenate([self.rows, self.rows[:1]]), 2)

    def test_missing_annotation_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, "every expected"):
            validate_ground_truth(self.rows[self.rows[:, 0] == 1], 2)

    def test_unknown_class_rejected(self):
        self.rows[0, 7] = 7
        with self.assertRaisesRegex(ValueError, "Unknown CroHD"):
            validate_ground_truth(self.rows, 2)

    def test_nonfinite_ground_truth_rejected(self):
        self.rows[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_ground_truth(self.rows, 2)


if __name__ == "__main__":
    unittest.main()
