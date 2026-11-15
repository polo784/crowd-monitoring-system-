"""Focused offline contracts for model reuse, raw counts, playback and stale work."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from crowd_inference import DensityPrediction
from dashboard import create_app, inspect_video, prepare_playback
from dashboard_service import DashboardRuntime, RuntimeProblem, attention_level, compatible_size, heatmap_png, load_config


class FakePredictor:
    calls = 0

    def __init__(self, checkpoint=None, device=None):
        type(self).calls += 1
        self.metadata = {"checkpoint_epoch": 16, "device_name": "CPU test", "normalization": "imagenet"}
        self.entered, self.release = None, None

    def predict(self, frame):
        if self.entered:
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("Test release timed out")
        density = np.array([[2, -3], [4, 1]], dtype=np.float32)
        return DensityPrediction(density, float(density.sum(dtype=np.float64)),
                                 (1, 3, *frame.shape[:2]), (1, 1, 2, 2), 1, 10, 1)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.config = load_config()
        self.config["history_limit"] = 2
        self.runtime = DashboardRuntime(self.config, FakePredictor)
        self.runtime.predictor, self.runtime.model_state = FakePredictor(), "ready"
        self.metadata = {"height": 32, "width": 48, "fps": 25, "duration": 2,
                         "model_size_hw": (32, 48), "reported_frames": 50, "codec": "h264"}
        self.session = self.runtime.select_video(self.root / "input.mp4", "input.mp4", self.metadata)
        ok, encoded = cv2.imencode(".jpg", np.full((32, 48, 3), 100, np.uint8))
        self.assertTrue(ok)
        self.frame = encoded.tobytes()

    def tearDown(self):
        self.runtime.close()
        self.scratch.cleanup()

    def play(self, epoch=1):
        self.runtime.control(self.session["id"], epoch, "play", 0, True)

    def infer(self, timestamp=.1, epoch=1, snapshot=False):
        return self.runtime.infer(self.session["id"], epoch, timestamp, self.frame, snapshot)

    def test_attention_boundaries(self):
        t = {"watch": 40, "elevated": 70, "review": 100}
        self.assertEqual([attention_level(n, t) for n in (-1, 39.9, 40, 70, 100)],
                         ["UNAVAILABLE", "NORMAL", "WATCH", "ELEVATED", "REVIEW"])
        with self.assertRaises(ValueError):
            attention_level(float("nan"), t)

    def test_stride_resize_and_pixel_budget(self):
        self.assertEqual(compatible_size(540, 960, 2073600), (544, 960))
        self.assertEqual(compatible_size(480, 640, 2073600), (480, 640))
        h, w = compatible_size(2160, 3840, 2073600)
        self.assertEqual(h % 8 + w % 8, 0)
        self.assertLessEqual(h * w, 2073600)
        self.assertAlmostEqual(h / w, 2160 / 3840, delta=.01)

    def test_invalid_config_rejected(self):
        config = copy.deepcopy(self.config)
        config["attention_thresholds"]["watch"] = 1000
        path = self.root / "bad.json"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "increasing"):
            load_config(path)

    def test_raw_signed_count_not_visualized_count(self):
        self.play()
        result = self.infer()
        self.assertEqual(result["count"], 4)
        self.assertNotEqual(result["count"], float(np.maximum(self.runtime.session.latest_density, 0).sum()))
        self.assertTrue(result["heatmap"].startswith("data:image/png;base64,"))
        np.testing.assert_array_equal(self.runtime.session.latest_density, [[2, -3], [4, 1]])
        self.assertEqual(result["timestamp"], .1)

    def test_display_does_not_mutate_density(self):
        density = np.array([[0, -3], [4, 1]], np.float32)
        original = density.copy()
        heatmap_png(density)
        np.testing.assert_array_equal(density, original)

    def test_pause_preserves_history_but_rejects_pending_sample(self):
        self.play(); self.infer()
        self.runtime.control(self.session["id"], 2, "pause", .2, False)
        self.assertEqual(len(self.runtime.session.history), 1)
        with self.assertRaisesRegex(RuntimeProblem, "Stale"):
            self.infer(.2, epoch=1)
        with self.assertRaisesRegex(RuntimeProblem, "paused"):
            self.infer(.2, epoch=2)

    def test_seek_and_restart_clear_history_and_density(self):
        self.play(); self.infer()
        self.runtime.control(self.session["id"], 2, "seek", 1, False)
        self.assertIsNone(self.runtime.session.latest)
        self.assertIsNone(self.runtime.session.latest_density)
        self.assertEqual(list(self.runtime.session.history), [])
        self.infer(1, epoch=2, snapshot=True)
        self.runtime.control(self.session["id"], 3, "restart", 0, False)
        self.assertEqual(list(self.runtime.session.history), [])
        self.assertEqual(self.runtime.session.timestamp, 0)

    def test_seek_during_inference_discards_late_gpu_result(self):
        self.play()
        predictor = self.runtime.predictor
        predictor.entered, predictor.release = threading.Event(), threading.Event()
        errors = []
        def work():
            try:
                self.infer()
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=work)
        worker.start()
        self.assertTrue(predictor.entered.wait(3))
        self.runtime.control(self.session["id"], 2, "seek", 1, True)
        predictor.release.set(); worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors[0].status, 409)
        self.assertIsNone(self.runtime.session.latest)
        self.assertEqual(len(self.runtime.session.history), 0)

    def test_busy_model_does_not_queue_frames(self):
        self.play()
        self.runtime.inference_lock.acquire()
        try:
            with self.assertRaises(RuntimeProblem) as error:
                self.infer()
            self.assertEqual(error.exception.status, 429)
        finally:
            self.runtime.inference_lock.release()

    def test_repeated_sample_and_old_control_rejected(self):
        self.play(); self.infer()
        with self.assertRaisesRegex(RuntimeProblem, "Repeated"):
            self.infer()
        with self.assertRaisesRegex(RuntimeProblem, "Old playback"):
            self.runtime.control(self.session["id"], 1, "pause", .2, False)

    def test_history_is_timestamped_and_bounded(self):
        self.play()
        for timestamp in (.1, .2, .3):
            self.infer(timestamp)
        self.assertEqual([p["timestamp"] for p in self.runtime.session.history], [.2, .3])

    def test_malformed_numeric_config(self):
        for key, value in (("history_limit", 1.5), ("history_limit", True), ("history_limit", 0),
                           ("history_limit", 1201), ("max_sample_hz", "4"), ("max_sample_hz", float("inf")),
                           ("max_model_pixels", True), ("health_timeout_seconds", 0), ("device", "gpu"),
                           ("checkpoint_sha256", "oops")):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.config); bad[key] = value
                path = self.root / "bad.json"; path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, key): load_config(path)
        for value in (True, "40", None, float("nan")):
            bad = copy.deepcopy(self.config); bad["attention_thresholds"]["watch"] = value
            path = self.root / "bad.json"; path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "thresholds"): load_config(path)

    def test_export_precision_truncation_and_segment_isolation(self):
        self.play()
        old_segment = self.runtime.session.segment_id
        for stamp in (.123456789012345, .2, .3): self.infer(stamp)
        export = self.runtime.export_history()
        self.assertEqual(len(export["observations"]), 2)
        self.assertEqual(export["metadata"]["discarded_observations"], 1)
        self.assertEqual(export["metadata"]["retained_range_seconds"], [.2, .3])
        row = export["observations"][0]
        for key in ("count", "generation", "attention", "segment_id", "input_shape", "output_shape",
                    "inference_ms", "service_ms", "capture_time_utc", "observation_time_utc"):
            self.assertIn(key, row)
        self.runtime.control(self.session["id"], 2, "seek", 1, False)
        self.assertNotEqual(self.runtime.session.segment_id, old_segment)
        self.assertEqual(self.runtime.export_history()["observations"], [])
        self.infer(1, epoch=2, snapshot=True)
        self.assertEqual(self.runtime.export_history()["metadata"]["discarded_observations"], 0)

    def test_cancelled_client_cannot_queue_gpu_and_health_responds(self):
        self.play()
        predictor = self.runtime.predictor
        predictor.entered, predictor.release = threading.Event(), threading.Event()
        errors = []
        def work():
            try: self.infer()
            except RuntimeProblem as error: errors.append(error)
        worker = threading.Thread(target=work); worker.start()
        self.assertTrue(predictor.entered.wait(3))
        # Simulate browser cancellation + invalidation while actual work continues.
        self.runtime.control(self.session["id"], 2, "invalidate", .1, False)
        with self.assertRaises(RuntimeProblem) as error: self.infer(.2, epoch=2, snapshot=True)
        self.assertEqual(error.exception.status, 429)
        self.assertTrue(self.runtime.status()["inference_busy"])
        with TestClient(create_app(self.config, self.runtime)) as client:
            self.assertEqual(client.get("/api/status").status_code, 200)
        predictor.release.set(); worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(self.runtime.session) # TestClient clean shutdown removed it.
        self.assertEqual(errors[0].status, 409)

    def test_oversized_nonfinite_and_oom_inputs(self):
        self.play()
        with self.assertRaises(RuntimeProblem) as error:
            self.runtime.infer(self.session["id"], 1, .1, b"x"*(self.config["max_frame_bytes"]+1))
        self.assertEqual(error.exception.status, 413)
        with patch.object(self.runtime.predictor, "predict", return_value=DensityPrediction(
                np.array([[float("nan")]], np.float32), float("nan"), (1,3,32,48), (1,1,1,1), 0,1,0)):
            with self.assertRaisesRegex(RuntimeProblem, "non-finite"): self.infer()
        import torch
        with patch.object(self.runtime.predictor, "predict", side_effect=torch.cuda.OutOfMemoryError("fixture")):
            with self.assertRaisesRegex(RuntimeProblem, "No automatic resize retry"): self.infer()
        self.assertFalse(self.runtime.inference_lock.locked())
        self.assertIsNone(self.runtime.session.latest)

    def test_real_codec_conversion_preserves_original_metadata(self):
        import imageio_ffmpeg
        path = self.root / "odd.mkv"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-f", "lavfi", "-i", "testsrc=size=49x33:rate=5", "-t", "0.6",
                        "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)], check=True, capture_output=True, timeout=30)
        metadata = inspect_video(path, self.config)
        self.assertEqual((metadata["height"], metadata["width"]), (33,49))
        with tempfile.TemporaryDirectory() as folder:
            class Scratch: name = folder
            playback = prepare_playback(path, Scratch(), metadata, self.config)
            self.assertTrue(playback.is_file())
            self.assertEqual((metadata["original"]["height"],metadata["original"]["width"]), (33,49))
            self.assertEqual((metadata["playback"]["height"],metadata["playback"]["width"]), (34,50))
            self.assertEqual(metadata["conversion"]["padding_hw"], [1,1])
            self.assertTrue(metadata["conversion"]["audio_removed"])
            self.assertEqual(metadata["conversion"]["crf"], 18)

    def test_replacing_video_invalidates_old_session(self):
        self.runtime.select_video(self.root / "new.mp4", "new.mp4", self.metadata)
        with self.assertRaisesRegex(RuntimeProblem, "replaced"):
            self.infer()

    def test_timestamp_and_frame_errors(self):
        self.play()
        for timestamp in (-1, float("nan"), 3):
            with self.assertRaisesRegex(RuntimeProblem, "timestamp"):
                self.infer(timestamp)
        with self.assertRaisesRegex(RuntimeProblem, "invalid"):
            self.runtime.infer(self.session["id"], 1, .1, b"bad")

    def test_checkpoint_load_once_and_missing_checkpoint_status(self):
        checkpoint = self.root / "fake.pt"; checkpoint.write_bytes(b"checkpoint fixture")
        config = copy.deepcopy(self.config); config["checkpoint"] = str(checkpoint)
        config["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        FakePredictor.calls = 0
        runtime = DashboardRuntime(config, FakePredictor)
        with TestClient(create_app(config, runtime)) as client:
            # Startup loading runs in the background; wait through its existing thread.
            import time
            for _ in range(50):
                state = client.get("/api/status").json()
                if state["model_state"] == "ready": break
                time.sleep(.01)
            self.assertEqual(state["model"]["checkpoint_epoch"], 16)
            self.assertEqual(FakePredictor.calls, 1)
        config["checkpoint"] = str(self.root / "missing.pt")
        broken = DashboardRuntime(config, FakePredictor)
        with self.assertLogs("crowd.dashboard", level="ERROR"):
            broken.load_model()
        self.assertEqual(broken.status()["model_state"], "error")

    def test_api_controls_inference_invalid_video_and_origin(self):
        with TestClient(create_app(self.config, self.runtime)) as client:
            command = dict(session_id=self.session["id"], generation=1, action="play", timestamp=0, playing=True)
            self.assertEqual(client.post("/api/control", json=command).status_code, 200)
            result = client.post("/api/infer", data=dict(session_id=self.session["id"], generation=1, timestamp=.1),
                                 files={"frame": ("frame.jpg", self.frame, "image/jpeg")})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["count"], 4)
            self.assertEqual(client.post("/api/video", files={"file": ("bad.txt", b"no")}).status_code, 422)
            self.assertEqual(client.post("/api/video", files={"file": ("bad.mp4", b"no")}).status_code, 422)
            self.assertEqual(client.post("/api/control", json=command, headers={"Origin":"https://example.com"}).status_code, 403)

    def test_upload_and_range_playback(self):
        path = self.root / "fixture.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (48, 32))
        for _ in range(3): writer.write(np.full((32, 48, 3), 50, np.uint8))
        writer.release()
        with patch("dashboard.prepare_playback", side_effect=lambda path, scratch, metadata, config: path):
            with TestClient(create_app(self.config, self.runtime)) as client:
                response = client.post("/api/video", files={"file": ("fixture.mp4", path.read_bytes(), "video/mp4")})
                self.assertEqual(response.status_code, 200)
                info = response.json(); self.assertEqual(info["video"]["fps"], 25)
                playback = client.get(info["video_url"], headers={"Range":"bytes=0-15"})
                self.assertEqual(playback.status_code, 206)
                self.assertEqual(len(playback.content), 16)


if __name__ == "__main__":
    unittest.main()
