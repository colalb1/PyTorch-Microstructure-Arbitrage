import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)

        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.act1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.act2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.act = nn.GELU()
        self.init_weights()

    def init_weights(self):
        # Kaiming init, usually better than ReLU/GeLU
        # https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
        nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.act(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()

        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        # Kaiming init final layer weights
        nn.init.kaiming_normal_(
            self.linear.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

        # Init bias to zero
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Input shape: (N, C, L)
        y1 = self.tcn(x)  # Output shape: (N, C_out, L)

        # Global Average Pooling pools across the entire sequence. This aggregates
        # features and is more robust to noise
        y_pooled = torch.nn.functional.adaptive_avg_pool1d(y1, 1).squeeze(
            -1
        )  # Shape: (N, C_out)

        return self.linear(y_pooled)
