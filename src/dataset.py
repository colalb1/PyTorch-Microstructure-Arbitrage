import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features are expected to be (num_samples, seq_len, num_features)
        # The model expects (N, C, L) -> (batch_size, num_features, seq_len) \implies
        # permute dimensions
        return torch.tensor(self.features[idx], dtype=torch.float32).permute(
            1, 0
        ), torch.tensor(self.labels[idx], dtype=torch.long)


def create_data_loaders(batch_size, data_path="data/processed"):
    """
    Creates training, validation, and test data loaders.
    """
    X_train = torch.load(f"{data_path}/X_train.pt")
    y_train = torch.load(f"{data_path}/y_train.pt")

    X_val = torch.load(f"{data_path}/X_val.pt")
    y_val = torch.load(f"{data_path}/y_val.pt")

    X_test = torch.load(f"{data_path}/X_test.pt")
    y_test = torch.load(f"{data_path}/y_test.pt")

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # Shuffle is false for time series
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
