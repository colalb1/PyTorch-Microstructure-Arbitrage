from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_config
from src.dataset import create_data_loaders
from src.model import TCNModel
from src.utils import plot_training_curves, save_model


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): The DataLoader for training data.
        loss_fn (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to train on.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.train()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for features, labels in tqdm(data_loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The model to validate.
        data_loader (DataLoader): The DataLoader for validation data.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device to validate on.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def main() -> None:
    """
    Main function to run the training and validation process.
    """
    # Setup
    device = torch.device(model_config.DEVICE)
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, _ = create_data_loaders(
        model_config.BATCH_SIZE, model_config.PROCESSED_DATA_DIR
    )

    # Model
    model = TCNModel(
        input_size=model_config.INPUT_SIZE,
        output_size=model_config.OUTPUT_SIZE,
        num_channels=model_config.NUM_CHANNELS,
        kernel_size=model_config.KERNEL_SIZE,
        dropout=model_config.DROPOUT,
    ).to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_config.LEARNING_RATE,
        weight_decay=model_config.WEIGHT_DECAY,
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("Starting training...")
    for epoch in range(model_config.NUM_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{model_config.NUM_EPOCHS} ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_config.MODEL_SAVE_PATH)

            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print("Training finished.")

    # Plotting
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)


if __name__ == "__main__":
    main()
