import logging
import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import model_config

logger = logging.getLogger(__name__)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
) -> None:
    """
    Plots and saves the training and validation loss and accuracy curves.

    Args:
        train_losses (List[float]): A list of training losses.
        val_losses (List[float]): A list of validation losses.
        train_accuracies (List[float]): A list of training accuracies.
        val_accuracies (List[float]): A list of validation accuracies.
    """
    if not os.path.exists(model_config.PLOTS_DIR):
        os.makedirs(model_config.PLOTS_DIR)

    epochs = range(1, len(train_losses) + 1)

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Training and Validation Metrics", fontsize=16)

    # Loss plot
    ax1.plot(epochs, train_losses, "o-", label="Training Loss")
    ax1.plot(epochs, val_losses, "o-", label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accuracies, "o-", label="Training Accuracy")
    ax2.plot(epochs, val_accuracies, "o-", label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    save_path = model_config.PLOT_SAVE_PATH
    plt.savefig(save_path)
    logger.info(f"Training plot saved to {save_path}")
    plt.close()


def save_model(model: nn.Module, path: str) -> None:
    """
    Saves the model state dictionary.

    Args:
        model (nn.Module): The model to save.
        path (str): The path to save the model to.
    """

    if not os.path.exists(model_config.MODELS_DIR):
        os.makedirs(model_config.MODELS_DIR)

    torch.save(model.state_dict(), path)

    logger.info(f"Model saved to {path}")
