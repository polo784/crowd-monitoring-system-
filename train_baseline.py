"""Train the tiny Mall crowd-density CNN for one pipeline smoke-test epoch."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from crowd_model import CrowdDensityCNN
from mall_dataloader import test_loader, train_loader, validation_loader


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "crowd_density_cnn_smoke_test.pt"
EPOCHS = 1
LEARNING_RATE = 1e-4


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, bool, bool]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    backprop_verified = False
    optimizer_step_verified = False

    for batch_number, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {predictions.shape} does not match target shape {targets.shape}"
            )

        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_number}")

        parameters_before_step = None
        if not optimizer_step_verified:
            parameters_before_step = [
                parameter.detach().clone()
                for parameter in model.parameters()
                if parameter.requires_grad
            ]

        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"Missing or non-finite gradients at batch {batch_number}")
        backprop_verified = True

        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"Non-finite model parameter at batch {batch_number}")

        if parameters_before_step is not None:
            parameters_after_step = [
                parameter.detach()
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            optimizer_step_verified = any(
                not torch.equal(before, after)
                for before, after in zip(parameters_before_step, parameters_after_step)
            )
            if not optimizer_step_verified:
                raise RuntimeError("Adam completed a step but no model parameter changed")

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_number % 50 == 0 or batch_number == len(loader):
            print(f"Training batch {batch_number}/{len(loader)}", flush=True)

    return total_loss / total_samples, backprop_verified, optimizer_step_verified


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_absolute_count_error = 0.0
    total_squared_count_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_number, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)

            if predictions.shape != targets.shape:
                raise RuntimeError(
                    f"Prediction shape {predictions.shape} does not match target shape {targets.shape}"
                )
            if not torch.isfinite(predictions).all():
                raise FloatingPointError(f"Non-finite {split_name.lower()} prediction")

            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {split_name.lower()} loss")

            predicted_counts = predictions.sum(dim=(1, 2, 3))
            target_counts = targets.sum(dim=(1, 2, 3))
            count_errors = predicted_counts - target_counts

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_absolute_count_error += count_errors.abs().sum().item()
            total_squared_count_error += count_errors.square().sum().item()
            total_samples += batch_size

            if batch_number % 25 == 0 or batch_number == len(loader):
                print(f"{split_name} batch {batch_number}/{len(loader)}", flush=True)

    average_loss = total_loss / total_samples
    mae = total_absolute_count_error / total_samples
    rmse = math.sqrt(total_squared_count_error / total_samples)
    return average_loss, mae, rmse


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    # Seed 1 avoids an all-zero final-ReLU initialization in this smoke-test model.
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowdDensityCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)

    training_loss = 0.0
    backprop_verified = False
    optimizer_step_verified = False
    for epoch in range(1, EPOCHS + 1):
        training_loss, backprop_verified, optimizer_step_verified = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Epoch {epoch} average training loss: {training_loss:.8f}", flush=True)

    validation_loss, validation_mae, validation_rmse = evaluate(
        model, validation_loader, criterion, device, "Validation"
    )
    test_loss, test_mae, test_rmse = evaluate(
        model, test_loader, criterion, device, "Test"
    )

    metrics = {
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("One or more final metrics are non-finite")

    save_checkpoint(CHECKPOINT_PATH, model, optimizer, metrics)

    print(f"Validation loss: {validation_loss:.8f}")
    print(f"Validation MAE:  {validation_mae:.6f}")
    print(f"Validation RMSE: {validation_rmse:.6f}")
    print(f"Test MAE:        {test_mae:.6f}")
    print(f"Test RMSE:       {test_rmse:.6f}")
    print(f"Backpropagation verified: {backprop_verified}")
    print(f"Optimizer step verified:  {optimizer_step_verified}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
