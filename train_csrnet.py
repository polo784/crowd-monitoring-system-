"""Train CSRNet on the existing Mall splits; select by validation count MAE.

Fresh run: uv run python train_csrnet.py --epochs 1 --lr 1e-5
Resume: uv run python train_csrnet.py --epochs 10 --lr 1e-5 --resume CHECKPOINT
No Dataset, image file, annotation, density file, or split is modified.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader

from csrnet import CSRNet
from density_utils import density_counts, resize_density_preserving_count
# Only actual training loads the Mall files; importing helpers/CI needs no data.
train_loader = validation_loader = test_loader = None


PROJECT_ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def atomic_write_json(path: Path, value: dict) -> None:
    """Replace a JSON file only after its complete new contents are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_save_checkpoint(payload: dict, path: Path) -> None:
    """Keep the previous checkpoint if writing its replacement fails."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    validation: dict[str, float],
    config: dict,
    history: list[dict],
    best_epoch: int,
    best_mae: float,
) -> dict:
    """Collect everything needed to resume after a completed epoch."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation["loss"],
        "validation_mae": validation["mae"],
        "validation_rmse": validation["rmse"],
        "training_config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def validate_resume_config(saved: dict, current: dict) -> None:
    """Refuse to mix a checkpoint with a different training setup."""
    keys = (
        "model", "learning_rate", "optimizer", "loss", "pretrained_vgg16",
        "weights", "seed", "batch_size", "parameters", "splits", "input_resolution",
        "output_resolution",
    )
    differences = [
        key for key in keys
        if key in saved and key in current and saved[key] != current[key]
    ]
    if differences:
        raise ValueError(
            "Resume checkpoint does not match the current configuration: "
            + ", ".join(differences)
        )


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> tuple[int, int, float, list[dict]]:
    """Restore model, optimizer, history, and RNG state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_resume_config(checkpoint.get("training_config", {}), config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = int(checkpoint["epoch"])
    best_epoch = int(checkpoint.get("best_epoch", completed_epoch))
    best_mae = float(
        checkpoint.get("best_validation_mae", checkpoint["validation_mae"])
    )
    history = list(checkpoint.get("history", []))
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    else:
        LOG.warning("Older checkpoint has no saved RNG state; continuation is valid but not exact.")
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )
    return completed_epoch + 1, best_epoch, best_mae, history


def normalize_images(images: torch.Tensor, pretrained: bool) -> torch.Tensor:
    """Only ImageNet channel normalization; no classification resize or crop."""
    if not pretrained:
        return images
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def predict_and_match(
    model: nn.Module, images: torch.Tensor, targets: torch.Tensor, pretrained: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = model(normalize_images(images, pretrained))
    resized = resize_density_preserving_count(targets, tuple(predictions.shape[-2:]))
    if predictions.shape != resized.shape:
        raise RuntimeError(f"Prediction {predictions.shape} != target {resized.shape}")
    return predictions, resized


def count_error(original: torch.Tensor, resized: torch.Tensor) -> float:
    return float((density_counts(original) - density_counts(resized)).abs().max().item())


def preflight(model: nn.Module, device: torch.device, pretrained: bool) -> dict:
    """Inspect a real full-size training batch before allowing optimization."""
    images, targets = next(iter(train_loader))
    images, targets = images.to(device), targets.to(device)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        predictions, resized = predict_and_match(model, images, targets, pretrained)
    if not torch.isfinite(predictions).all():
        raise FloatingPointError("Non-finite preflight prediction")
    error = count_error(targets, resized)
    details = {
        "input_shape": list(images.shape),
        "original_target_shape": list(targets.shape),
        "prediction_shape": list(predictions.shape),
        "resized_target_shape": list(resized.shape),
        "original_counts": density_counts(targets).tolist(),
        "resized_counts": density_counts(resized).tolist(),
        "max_count_error": error,
        "forward_seconds": time.perf_counter() - started,
    }
    for name, value in details.items():
        LOG.info("Preflight %s: %s", name, value)
    return details


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: nn.Module, device: torch.device, pretrained: bool,
) -> dict[str, float]:
    model.train()
    total_loss, samples, max_error = 0.0, 0, 0.0
    started = time.perf_counter()
    for step, (images, original_targets) in enumerate(loader, 1):
        images, original_targets = images.to(device), original_targets.to(device)
        predictions, targets = predict_and_match(model, images, original_targets, pretrained)
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        size = images.shape[0]
        total_loss += loss.item() * size
        samples += size
        max_error = max(max_error, count_error(original_targets, targets))
        if step == 1 or step % 10 == 0 or step == len(loader):
            elapsed = time.perf_counter() - started
            LOG.info(
                "Train %d/%d | mean MSE %.8f | %.2f s/batch | estimated train %.1f min",
                step, len(loader), total_loss / samples, elapsed / step,
                elapsed / step * len(loader) / 60,
            )
    return {"loss": total_loss / samples, "max_count_error": max_error,
            "seconds": time.perf_counter() - started}


def save_prediction(
    image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor,
    true_count: float, path: Path,
) -> None:
    """Three plain panels; low-resolution maps are enlarged for DISPLAY only."""
    rgb = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    gt = target.detach().cpu().squeeze(0).numpy()
    pred = prediction.detach().cpu().squeeze(0).numpy()
    limit = max(float(np.abs(gt).max()), float(np.abs(pred).max()), 1e-12)
    height, width = rgb.shape[:2]

    def heatmap(values: np.ndarray) -> Image.Image:
        scaled = np.clip(values / limit, -1, 1)
        colors = np.zeros((*values.shape, 3), dtype=np.uint8)
        colors[..., 0] = (255 * np.maximum(scaled, 0)).astype(np.uint8)
        colors[..., 2] = (255 * np.maximum(-scaled, 0)).astype(np.uint8)
        return Image.fromarray(colors).resize((width, height), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (3 * width, height + 64), "white")
    for index, panel in enumerate((Image.fromarray(rgb), heatmap(gt), heatmap(pred))):
        canvas.paste(panel, (index * width, 64))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    predicted_count = float(prediction.sum(dtype=torch.float64).item())
    labels = (
        "Original RGB image",
        f"Ground truth count: {true_count:.2f}",
        f"Predicted count: {predicted_count:.2f}",
    )
    for index, label in enumerate(labels):
        draw.text((index * width + 10, 7), label, fill="black", font=font)
    draw.text((width + 10, 34), f"Density @ {gt.shape}: shared scale", fill="black", font=font)
    draw.text((2 * width + 10, 34), "Red: positive | Blue: negative", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    pretrained: bool, name: str, prediction_dir: Path | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss, absolute_error, squared_error = 0.0, 0.0, 0.0
    samples, max_error = 0, 0.0
    with torch.no_grad():
        for step, (images, original_targets) in enumerate(loader, 1):
            images, original_targets = images.to(device), original_targets.to(device)
            predictions, targets = predict_and_match(model, images, original_targets, pretrained)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at batch {step}")
            # Evaluation ground truth ALWAYS comes from the original full-resolution map.
            true_counts = density_counts(original_targets)
            errors = density_counts(predictions) - true_counts
            size = images.shape[0]
            total_loss += loss.item() * size
            absolute_error += errors.abs().sum().item()
            squared_error += errors.square().sum().item()
            max_error = max(max_error, count_error(original_targets, targets))

            if prediction_dir is not None and samples < 3:
                for index in range(min(size, 3 - samples)):
                    save_prediction(images[index], targets[index], predictions[index],
                                    true_counts[index].item(),
                                    prediction_dir / f"test_sample_{samples + index + 1:03d}.png")
            samples += size
            if step == 1 or step % 25 == 0 or step == len(loader):
                LOG.info("%s %d/%d", name, step, len(loader))
    result = {"loss": total_loss / samples, "mae": absolute_error / samples,
              "rmse": math.sqrt(squared_error / samples), "max_count_error": max_error}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Non-finite {name} metrics")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Explicitly opt out of VGG16 weights and ImageNet normalization")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--resume", type=Path,
        help="Resume from a completed-epoch checkpoint; --epochs is the total epoch count",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Finish after validation; useful before the final test evaluation",
    )
    parser.add_argument("--require-cuda", action="store_true",
                        help="Stop instead of falling back to CPU training")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify real batch shapes/counts without training or test evaluation")
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    return args


def main() -> None:
    global train_loader, validation_loader, test_loader
    args = parse_args()
    if train_loader is None:
        from mall_dataloader import train_loader, validation_loader, test_loader
    checkpoint_path = args.output_dir / "checkpoints" / "csrnet_best.pt"
    latest_checkpoint_path = args.output_dir / "checkpoints" / "csrnet_latest.pt"
    report_path = args.output_dir / "csrnet_results.json"
    if args.resume is None and not args.preflight_only and any(
        path.exists() for path in (checkpoint_path, latest_checkpoint_path, report_path)
    ):
        raise FileExistsError("This output folder already contains a run. Use --resume or a new folder.")
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        if args.resume.parent != checkpoint_path.parent.resolve():
            raise ValueError("Resume checkpoint must be in --output-dir/checkpoints; copy the run first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.output_dir / f"csrnet_{run_id}.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True, handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8"),
    ])
    torch.manual_seed(args.seed)
    torch.hub.set_dir(str(args.output_dir / "torch_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run; select the Colab GPU kernel.")
    pretrained = not args.no_pretrained
    LOG.info("Device: %s | pretrained requested: %s | batch size: %s | lr: %g | epochs: %d",
             device, pretrained, train_loader.batch_size, args.lr, args.epochs)
    # A failed download raises an error; there is NO silent random-weight fallback.
    # A resumed checkpoint supplies the whole model, including its VGG frontend.
    model = CSRNet(pretrained=pretrained and args.resume is None).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    LOG.info("Pretrained VGG16 used: %s | trainable parameters: %d", pretrained, parameters)
    verified = preflight(model, device, pretrained)
    if args.preflight_only:
        LOG.info("Preflight passed; no training or test evaluation requested.")
        return

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "CSRNet", "epochs": args.epochs, "learning_rate": args.lr,
        "optimizer": "Adam", "loss": "MSELoss(mean)", "seed": args.seed,
        "pretrained_vgg16": pretrained,
        "weights": "VGG16_Weights.IMAGENET1K_V1" if pretrained else None,
        "imagenet_mean": [0.485, 0.456, 0.406] if pretrained else None,
        "imagenet_std": [0.229, 0.224, 0.225] if pretrained else None,
        "device": str(device), "batch_size": train_loader.batch_size,
        "parameters": parameters, "torch_version": str(torch.__version__),
        "splits": {"train": [1, 1400], "validation": [1401, 1700], "test": [1701, 2000]},
        "input_resolution": verified["input_shape"][-2:],
        "output_resolution": verified["prediction_shape"][-2:],
    }
    history: list[dict] = []
    best_epoch = 0
    best_mae = math.inf
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_epoch, best_mae, history = load_resume_checkpoint(
            resume_path, model, optimizer, device, config
        )
        # The original one-epoch checkpoint predates embedded history. Recover it
        # from the adjacent result JSON when resuming inside the same run folder.
        previous_report = resume_path.parent.parent / "csrnet_results.json"
        if not history and previous_report.is_file():
            history = [entry for entry in json.loads(
                previous_report.read_text(encoding="utf-8")
            ).get("epochs", []) if entry["epoch"] < start_epoch]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "The run's best checkpoint is missing. Resume in its original --output-dir."
            )
        saved_best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_resume_config(saved_best.get("training_config", {}), config)
        if saved_best["epoch"] >= start_epoch:
            raise ValueError("Best checkpoint is newer than the resume point; resume from that best checkpoint.")
        best_epoch = int(saved_best["epoch"])
        best_mae = float(saved_best["validation_mae"])
        del saved_best
        LOG.info(
            "Resumed completed epoch %d; continuing at epoch %d toward total epoch %d",
            start_epoch - 1, start_epoch, args.epochs,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}; "
            f"--epochs must be at least {start_epoch}."
        )

    report_path = args.output_dir / "csrnet_results.json"
    for epoch in range(start_epoch, args.epochs + 1):
        LOG.info("Epoch %d/%d", epoch, args.epochs)
        started = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, pretrained)
        validation = evaluate(model, validation_loader, criterion, device, pretrained, "Validation")
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation,
                        "epoch_seconds": seconds})
        LOG.info("Train Loss: %.8f | Validation Loss: %.8f | Validation MAE: %.6f | "
                 "Validation RMSE: %.6f | Epoch Time: %.1f s", train_metrics["loss"],
                 validation["loss"], validation["mae"], validation["rmse"], seconds)
        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            best_epoch = epoch
            atomic_save_checkpoint(
                checkpoint_payload(
                    epoch, model, optimizer, validation, config, history,
                    best_epoch, best_mae,
                ),
                checkpoint_path,
            )
            LOG.info("Saved best validation checkpoint: %s", checkpoint_path)

        atomic_save_checkpoint(
            checkpoint_payload(
                epoch, model, optimizer, validation, config, history,
                best_epoch, best_mae,
            ),
            latest_checkpoint_path,
        )
        LOG.info("Saved latest resumable checkpoint: %s", latest_checkpoint_path)
        atomic_write_json(
            report_path,
            {
                "status": "training_complete" if epoch == args.epochs else "training",
                "config": config,
                "preflight": verified,
                "epochs": history,
                "best_epoch": best_epoch,
                "test": None,
                "checkpoint_path": str(checkpoint_path),
                "latest_checkpoint_path": str(latest_checkpoint_path),
                "prediction_dir": None,
                "log_path": str(log_path),
            },
        )

    if args.skip_test:
        LOG.info("Training complete; test evaluation skipped as requested.")
        return

    # Exactly one final test pass, using weights chosen only by validation MAE.
    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    LOG.info("Loaded best epoch %d for the single test evaluation", best["epoch"])
    prediction_dir = args.output_dir / "csrnet_predictions"
    test = evaluate(model, test_loader, criterion, device, pretrained, "Test", prediction_dir)
    LOG.info("Test MSE: %.8f | Test MAE: %.6f | Test RMSE: %.6f",
             test["loss"], test["mae"], test["rmse"])
    report = {"status": "complete", "config": config, "preflight": verified, "epochs": history,
              "best_epoch": best["epoch"], "test": test,
              "checkpoint_path": str(checkpoint_path), "prediction_dir": str(prediction_dir),
              "latest_checkpoint_path": str(latest_checkpoint_path),
              "log_path": str(log_path)}
    atomic_write_json(report_path, report)
    LOG.info("Completed. Results: %s | Predictions: %s", report_path, prediction_dir)


if __name__ == "__main__":
    main()
