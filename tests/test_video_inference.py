"""Small contract tests for raw counts, preprocessing, MP4 output and failures."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from crowd_inference import CSRNetPredictor
from video_inference import draw_overlay, run_video


class InspectModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.last_input = None

    def forward(self, images):
        self.last_input = images.clone()
        # Deliberately signed output to detect incorrect clamping in the count.
        h, w = images.shape[-2] // 8, images.shape[-1] // 8
        out = torch.full((1, 1, h, w), 2.0, device=images.device)
        out[:, :, 0, 0] = -3.0
        return out + self.offset


class VideoInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "model.pt"
        torch.save({"model_state_dict": InspectModel().state_dict(), "epoch": 3,
                    "training_config": {"model": "CSRNet", "pretrained_vgg16": True,
                                        "imagenet_mean": [.485, .456, .406],
                                        "imagenet_std": [.229, .224, .225]}}, self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def predictor(self, **kwargs):
        with patch("crowd_inference.CSRNet", InspectModel):
            return CSRNetPredictor(self.checkpoint, device="cpu", **kwargs)

    def make_video(self, name="input.mp4", frames=3):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (48, 32))
        self.assertTrue(writer.isOpened())
        for i in range(frames):
            frame = np.full((32, 48, 3), 20 + i * 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_rgb_normalization_raw_count_and_visualization_separation(self):
        predictor = self.predictor()
        frame = np.full((32, 48, 3), [0, 128, 255], dtype=np.uint8)
        prediction = predictor.predict(frame)
        expected_rgb = torch.tensor([1., 128 / 255, 0.])
        expected = (expected_rgb - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(predictor.model.last_input[0, :, 0, 0], expected)
        self.assertEqual(prediction.input_shape, (1, 3, 32, 48))
        self.assertEqual(prediction.output_shape, (1, 1, 4, 6))
        self.assertEqual(prediction.count, 43.0)  # 23*2 - 3, not 46.
        original = prediction.density.copy()
        draw_overlay(frame, prediction.density, prediction.count, 0, .5, None)
        np.testing.assert_array_equal(prediction.density, original)
        self.assertGreater(prediction.inference_ms, 0)

    def test_explicit_size_and_native_stride_guard(self):
        frame = np.zeros((34, 50, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "stride 8"):
            self.predictor().predict(frame)
        resized = self.predictor(input_size=(32, 48)).predict(frame)
        self.assertEqual(resized.input_shape, (1, 3, 32, 48))

    def test_missing_metadata_requires_explicit_normalization(self):
        torch.save(InspectModel().state_dict(), self.checkpoint)
        with self.assertRaisesRegex(ValueError, "normalization metadata"):
            self.predictor()
        predictor = self.predictor(normalization="none")
        predictor.predict(np.full((32, 48, 3), 255, dtype=np.uint8))
        self.assertTrue((predictor.model.last_input == 1).all())

    def test_video_outputs_frames_timestamps_and_raw_counts(self):
        video = self.make_video()
        out = self.root / "run"
        summary = run_video(self.predictor(), video, out)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["processed_frames"], 3)
        self.assertEqual(summary["input_video"]["nominal_fps"], 5)
        with (out / "measurements.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["predicted_count"]) for row in rows], [43.] * 3)
        np.testing.assert_allclose([float(row["timestamp_seconds"]) for row in rows], [0, .2, .4])
        output = cv2.VideoCapture(str(out / "annotated.mp4"))
        decoded = 0
        while True:
            ok, frame = output.read()
            if not ok:
                break
            self.assertEqual(frame.shape[:2], (32, 48))
            decoded += 1
        output.release()
        self.assertEqual(decoded, 3)
        self.assertGreater(summary["timing"]["effective_inference_fps"], 0)
        with self.assertRaises(FileExistsError):
            run_video(self.predictor(), video, out)

    def test_unnormalized_null_statistics(self):
        torch.save({"model_state_dict": InspectModel().state_dict(),
                    "training_config": {"pretrained_vgg16": False,
                                        "imagenet_mean": None, "imagenet_std": None}}, self.checkpoint)
        predictor = self.predictor()
        self.assertIsNone(predictor.mean)
        result = predictor.predict(np.full((32, 48, 3), 128, np.uint8))
        self.assertEqual(result.count, 43.)
        torch.testing.assert_close(predictor.model.last_input[0, :, 0, 0], torch.full((3,), 128/255))

    def test_required_statistics_cannot_be_missing_or_null(self):
        for extra in ({}, {"imagenet_mean": None, "imagenet_std": None},
                      {"imagenet_mean": [0, 0, 0], "imagenet_std": [1, 0, 1]}):
            torch.save({"model_state_dict": InspectModel().state_dict(),
                        "training_config": {"pretrained_vgg16": True, **extra}}, self.checkpoint)
            with self.assertRaisesRegex(ValueError, "normalization"):
                self.predictor()

    def test_frame_limit_recorded(self):
        summary = run_video(self.predictor(), self.make_video(), self.root / "limited", max_frames=1)
        self.assertEqual(summary["scope"], "limited_run")
        self.assertEqual(summary["processed_frames"], 1)

    def test_nonfinite_prediction_stops_and_records_failure(self):
        predictor = self.predictor()
        predictor.model.offset.data.fill_(float("nan"))
        out = self.root / "bad_run"
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            run_video(predictor, self.make_video(), out)
        self.assertFalse((out / "annotated.mp4").exists())
        report = json.loads((out / "summary.json").read_text())
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
