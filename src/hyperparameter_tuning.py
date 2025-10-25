import logging

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

import model_config
from src.dataset import create_data_loaders
from src.model import TCNModel
from src.train import train_epoch, validate_epoch

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): A single trial of the optimization.

    Returns:
        float: The best validation accuracy achieved in the trial.
    """
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 5)

    num_channels = []

    for i in range(num_layers):
        out_channels = trial.suggest_int(f"n_channels_l{i}", 16, 128, log=True)
        num_channels.append(out_channels)

    # Static params
    device = torch.device(model_config.DEVICE)

    train_loader, val_loader, _ = create_data_loaders(
        model_config.BATCH_SIZE, model_config.PROCESSED_DATA_DIR
    )

    model = TCNModel(
        input_size=model_config.INPUT_SIZE,
        output_size=model_config.OUTPUT_SIZE,
        num_channels=num_channels,
        kernel_size=model_config.KERNEL_SIZE,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=model_config.WEIGHT_DECAY
    )

    # Training
    # Smaller number of epochs for faster search
    best_val_acc = 0.0
    num_epochs = 15

    for _ in range(num_epochs):
        train_epoch(model, train_loader, loss_fn, optimizer, device)
        _, val_acc = validate_epoch(model, val_loader, loss_fn, device)

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


def main():
    """
    Runs Optuna hyperparameter tuning
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1 hour

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()
