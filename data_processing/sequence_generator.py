import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def simulate_feature_data(
    num_rows: int = 50000, num_features: int = 16
) -> pd.DataFrame:
    """
    Generates a dummy Pandas DataFrame to simulate feature data.

    Args:
        num_rows (int): The number of rows for the DataFrame.
        num_features (int): The total number of feature columns.

    Returns:
        pd.DataFrame: A DataFrame with a nanosecond timestamp index, 'cross_basis',
                      and other numerical feature columns.
    """
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    timestamps = pd.to_datetime(
        np.arange(start_time.value, start_time.value + num_rows * 1_000_000, 1_000_000)
    )

    data = np.random.randn(num_rows, num_features)
    columns = ["feature_" + str(i) for i in range(num_features - 1)] + ["cross_basis"]

    df = pd.DataFrame(data, index=timestamps, columns=columns)

    # Make 'cross_basis' a bit more like a real spread
    df["cross_basis"] = np.cumsum(np.random.randn(num_rows) * 1e-5) + 0.001

    print(f"Generated dummy data with {num_rows} rows and {num_features} features.")
    return df


def create_targets(
    df: pd.DataFrame, horizon_ms: int = 500, threshold: float = 1e-5
) -> np.ndarray:
    """
    Generates classification targets based on the future change in 'cross_basis'.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'cross_basis' column.
        horizon_ms (int): The time horizon in milliseconds to look into the future.
        threshold (float): The threshold for defining a 'STABLE' change.

    Returns:
        np.ndarray: A 1D NumPy array of integer class labels (0, 1, or 2).
    """
    df_shifted = df["cross_basis"].shift(-int(horizon_ms))
    delta = df_shifted - df["cross_basis"]

    y = np.full(len(df), 1, dtype=int)  # Default to STABLE (Class 1)

    y[delta > threshold] = 2  # WIDEN (Class 2)
    y[delta < -threshold] = 0  # TIGHTEN (Class 0)

    # The last 'horizon_ms' values will be NaN, so we keep them as STABLE or drop them
    # For simplicity, we'll keep them as STABLE, but in a real scenario, they should be handled carefully.
    print("Target vector 'y' created.")
    return y


def create_sequences(
    features: np.ndarray, targets: np.ndarray, sequence_length: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms flat feature/target arrays into 3D tensors using a sliding window.

    Args:
        features (np.ndarray): The 2D feature array.
        targets (np.ndarray): The 1D target array.
        sequence_length (int): The length of each sequence.

    Returns:
        tuple: A tuple containing the 3D feature tensor (X) and 1D target tensor (y).
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - sequence_length):
        X_seq.append(features[i : i + sequence_length])
        y_seq.append(
            targets[i + sequence_length - 1]
        )  # Target corresponds to the end of the sequence

    print(f"Created {len(X_seq)} sequences of length {sequence_length}.")
    return np.array(X_seq), np.array(y_seq)


def main():
    """
    Main function to run the data processing pipeline.
    """
    # Input Simulation
    df = simulate_feature_data(num_rows=55000, num_features=20)

    # Target Definition
    y_flat = create_targets(df, horizon_ms=500, threshold=1e-5)
    X_flat = df.values

    # Sequence Creation
    sequence_length = 100
    X_sequences, y_sequences = create_sequences(X_flat, y_flat, sequence_length)

    # Chronological Splitting
    total_samples = len(X_sequences)
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)

    X_train = X_sequences[:train_end]
    y_train = y_sequences[:train_end]

    X_val = X_sequences[train_end:val_end]
    y_val = y_sequences[train_end:val_end]

    X_test = X_sequences[val_end:]
    y_test = y_sequences[val_end:]

    print("\nData split chronologically:")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Convert to PyTorch Tensors
    X_train_pt = torch.tensor(X_train, dtype=torch.float32)
    y_train_pt = torch.tensor(y_train, dtype=torch.long)

    X_val_pt = torch.tensor(X_val, dtype=torch.float32)
    y_val_pt = torch.tensor(y_val, dtype=torch.long)

    X_test_pt = torch.tensor(X_test, dtype=torch.float32)
    y_test_pt = torch.tensor(y_test, dtype=torch.long)

    # Save data
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(X_train_pt, os.path.join(output_dir, "X_train.pt"))
    torch.save(y_train_pt, os.path.join(output_dir, "y_train.pt"))

    torch.save(X_val_pt, os.path.join(output_dir, "X_val.pt"))
    torch.save(y_val_pt, os.path.join(output_dir, "y_val.pt"))

    torch.save(X_test_pt, os.path.join(output_dir, "X_test.pt"))
    torch.save(y_test_pt, os.path.join(output_dir, "y_test.pt"))

    print(f"\nProcessed data saved to '{output_dir}' directory.")


if __name__ == "__main__":
    main()
