"""Focused offline tests for the CSRNet architecture and tensor interface."""

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from csrnet import CSRNet
from density_utils import density_counts, resize_density_preserving_count
from train_csrnet import (
    atomic_write_json,
    atomic_save_checkpoint,
    checkpoint_payload,
    evaluate,
    load_resume_checkpoint,
    normalize_images,
    save_prediction,
    train_one_epoch,
    validate_resume_config,
)
import train_csrnet


class DensityResizeTests(unittest.TestCase):
    def test_mall_resolution_is_block_sum(self):
        torch.manual_seed(0)
        target = torch.rand(4, 1, 480, 640) * 0.001
        resized = resize_density_preserving_count(target, (60, 80))
        expected = target.reshape(4, 1, 60, 8, 80, 8).sum(dim=(3, 5))
        torch.testing.assert_close(resized, expected)
        torch.testing.assert_close(density_counts(resized), density_counts(target),
                                   atol=1e-4, rtol=1e-6)

    def test_non_integer_ratio_edges_zero_and_tiny_maps(self):
        target = torch.zeros(4, 1, 17, 19)
        target[0, 0, 0, 0] = 2
        target[0, 0, -1, -1] = 3
        target[1] = 0.2
        target[3, 0, 8, 9] = 1e-10
        original = target.clone()
        resized = resize_density_preserving_count(target, (5, 7))
        torch.testing.assert_close(density_counts(resized), density_counts(target),
                                   atol=1e-12, rtol=1e-6)
        self.assertEqual(float(resized[2].sum()), 0)
        self.assertTrue(torch.equal(target, original))
        self.assertTrue((resized >= 0).all())

    def test_identity(self):
        target = torch.ones(2, 1, 8, 10)
        self.assertIs(resize_density_preserving_count(target, (8, 10)), target)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize_density_preserving_count(torch.ones(2, 8, 8), (4, 4))
        for value in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                resize_density_preserving_count(torch.full((1, 1, 8, 8), value), (4, 4))
        with self.assertRaises(ValueError):
            resize_density_preserving_count(torch.ones(1, 1, 8, 8), (9, 4))


class CSRNetTests(unittest.TestCase):
    def test_architecture_and_stride(self):
        model = CSRNet(pretrained=False).eval()
        self.assertEqual(sum(isinstance(x, nn.MaxPool2d) for x in model.modules()), 3)
        backend_convs = [x for x in model.backend if isinstance(x, nn.Conv2d)]
        self.assertEqual([x.out_channels for x in backend_convs], [512, 512, 512, 256, 128, 64])
        self.assertTrue(all(x.dilation == (2, 2) for x in backend_convs))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertIsInstance(model.output_layer, nn.Conv2d)
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 40))
        self.assertEqual(tuple(output.shape), (2, 1, 4, 5))

    def test_normalization_is_image_only_and_out_of_place(self):
        images = torch.zeros(2, 3, 8, 8)
        normalized = normalize_images(images, pretrained=True)
        expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(normalized[0, :, 0, 0], expected)
        self.assertEqual(float(images.sum()), 0)
        self.assertIs(normalize_images(images, pretrained=False), images)

    def test_count_metrics_and_no_evaluation_gradients(self):
        class ConstantDensity(nn.Module):
            def forward(self, images):
                return images.new_full((images.shape[0], 1, 1, 1), 6.0)

        images = torch.zeros(2, 3, 8, 8)
        targets = torch.ones(2, 1, 8, 8) / 64
        targets[0] *= 8
        targets[1] *= 4
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        metrics = evaluate(ConstantDensity(), loader, nn.MSELoss(), torch.device("cpu"), False, "Unit")
        self.assertAlmostEqual(metrics["loss"], 4.0)
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)

    def test_modular_training_step(self):
        torch.manual_seed(1)
        model = nn.Conv2d(3, 1, kernel_size=1, stride=8)
        before = model.weight.detach().clone()
        loader = DataLoader(TensorDataset(torch.ones(2, 3, 8, 8),
                                         torch.ones(2, 1, 8, 8) / 64), batch_size=2)
        metrics = train_one_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=1e-5),
                                  nn.MSELoss(), torch.device("cpu"), False)
        self.assertGreaterEqual(metrics["loss"], 0)
        self.assertFalse(torch.equal(before, model.weight))

    def test_prediction_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.png"
            save_prediction(torch.zeros(3, 64, 80), torch.ones(1, 8, 10),
                            torch.full((1, 8, 10), -0.5), 80.0, path)
            with Image.open(path) as saved:
                self.assertEqual(saved.size, (240, 128))

    def test_resume_checkpoint_round_trip(self):
        torch.manual_seed(7)
        model = nn.Conv2d(3, 1, kernel_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss = model(torch.ones(1, 3, 2, 2)).square().mean()
        loss.backward()
        optimizer.step()
        expected_weight = model.weight.detach().clone()
        config = {
            "model": "test", "learning_rate": 1e-5, "optimizer": "Adam",
            "loss": "MSELoss(mean)", "pretrained_vgg16": False,
            "batch_size": 4,
        }
        validation = {"loss": 0.5, "mae": 2.0, "rmse": 3.0}
        history = [{"epoch": 1}]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            torch.save(
                checkpoint_payload(
                    1, model, optimizer, validation, config, history, 1, 2.0
                ),
                checkpoint,
            )
            restored_model = nn.Conv2d(3, 1, kernel_size=1)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-5)
            start, best_epoch, best_mae, restored_history = load_resume_checkpoint(
                checkpoint, restored_model, restored_optimizer,
                torch.device("cpu"), config,
            )

        self.assertEqual((start, best_epoch, best_mae), (2, 1, 2.0))
        self.assertEqual(restored_history, history)
        torch.testing.assert_close(restored_model.weight, expected_weight)
        self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_resume_rejects_changed_configuration(self):
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_resume_config(
                {"model": "CSRNet", "learning_rate": 1e-5},
                {"model": "CSRNet", "learning_rate": 2e-5},
            )

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            atomic_write_json(path, {"status": "training", "epoch": 2})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "status": "training",\n  "epoch": 2\n}',
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_interrupted_checkpoint_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_save_checkpoint({"epoch": 1}, path)
            with patch("train_csrnet.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save_checkpoint({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)

    def test_training_resume_matches_uninterrupted_run(self):
        # Exercise the real main loop on tiny synthetic tensors. No Mall epoch runs.
        images = torch.arange(6 * 3 * 8 * 8, dtype=torch.float32).reshape(6, 3, 8, 8) / 1200
        targets = torch.ones(6, 1, 8, 8) / 64
        dataset = TensorDataset(images, targets)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        evaluation = DataLoader(dataset, batch_size=2)

        def run(directory, epochs, resume=None):
            argv = ["train_csrnet.py", "--no-pretrained", "--epochs", str(epochs),
                    "--output-dir", str(directory), "--skip-test"]
            if resume:
                argv += ["--resume", str(resume)]
            with patch("sys.argv", argv):
                train_csrnet.main()

        with tempfile.TemporaryDirectory() as directory, \
                patch("train_csrnet.CSRNet", side_effect=lambda **kwargs: nn.Conv2d(3, 1, 1, stride=8)), \
                patch("train_csrnet.train_loader", loader), \
                patch("train_csrnet.validation_loader", evaluation), \
                patch("train_csrnet.torch.cuda.is_available", return_value=False), \
                patch("train_csrnet.torch.hub.set_dir"):
            uninterrupted = Path(directory) / "uninterrupted"
            resumed = Path(directory) / "resumed"
            run(uninterrupted, 2)
            run(resumed, 1)
            run(resumed, 2, resumed / "checkpoints/csrnet_latest.pt")
            expected = torch.load(uninterrupted / "checkpoints/csrnet_latest.pt", weights_only=True)
            actual = torch.load(resumed / "checkpoints/csrnet_latest.pt", weights_only=True)
            for key in expected["model_state_dict"]:
                torch.testing.assert_close(expected["model_state_dict"][key],
                                           actual["model_state_dict"][key], rtol=0, atol=0)
            report = json.loads((resumed / "csrnet_results.json").read_text())
            self.assertEqual([entry["epoch"] for entry in report["epochs"]], [1, 2])
            self.assertIsNone(report["test"])
            with self.assertRaises(FileExistsError):
                run(resumed, 3)

            # The original Colab checkpoint has no RNG state or embedded history.
            legacy = Path(directory) / "legacy"
            run(legacy, 1)
            best_path = legacy / "checkpoints/csrnet_best.pt"
            old = torch.load(best_path, weights_only=True)
            for key in ("history", "torch_rng_state", "cuda_rng_state_all",
                        "best_epoch", "best_validation_mae"):
                old.pop(key, None)
            atomic_save_checkpoint(old, best_path)
            run(legacy, 2, best_path)
            history = json.loads((legacy / "csrnet_results.json").read_text())["epochs"]
            self.assertEqual([entry["epoch"] for entry in history], [1, 2])

            # main() installs a file logger; close it before Windows removes the test directory.
            import logging
            for handler in list(logging.getLogger().handlers):
                handler.close()
                logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
