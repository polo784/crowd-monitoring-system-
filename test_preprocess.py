import unittest

import numpy as np

from preprocess import generate_density_map


class GenerateDensityMapTests(unittest.TestCase):
    IMAGE_SHAPE = (480, 640)

    def assert_count_is_preserved(self, points):
        density_map = generate_density_map(self.IMAGE_SHAPE, points, sigma=15)
        actual_count = density_map.sum(dtype=np.float64)
        expected_count = len(points)
        self.assertTrue(
            np.isclose(actual_count, expected_count, atol=1e-5),
            f"Expected density sum {expected_count}, got {actual_count}",
        )

    def test_annotation_in_image_centre(self):
        self.assert_count_is_preserved([(320.0, 240.0)])

    def test_annotation_on_image_edge(self):
        self.assert_count_is_preserved([(0.0, 240.0)])

    def test_annotation_in_image_corner(self):
        self.assert_count_is_preserved([(0.0, 0.0)])

    def test_two_annotations_at_same_integer_pixel(self):
        self.assert_count_is_preserved(
            [
                (265.315983, 141.236067),
                (265.315983, 141.909043),
            ]
        )


if __name__ == "__main__":
    unittest.main()
