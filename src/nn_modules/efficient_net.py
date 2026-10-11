from torch import nn


class MBConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        expanded = channels * 2
        squeezed = max(1, channels // 4)

        self.features = nn.Sequential(
            nn.Conv1d(channels, expanded, 1, bias=False),
            nn.BatchNorm1d(expanded),
            nn.SiLU(),

            nn.Conv1d(
                expanded,
                expanded,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm1d(expanded),
            nn.SiLU(),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(expanded, squeezed, 1),
            nn.SiLU(),
            nn.Conv1d(squeezed, expanded, 1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv1d(expanded, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        y = self.features(x)
        y = y * self.se(y)
        return x + self.project(y)


class EfficientNet(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: int = 48,
        output_channels: int = 128,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_features, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),

            *[
                MBConv1d(channels, dilation=d)
                for d in (1, 2, 4, 8, 16)
            ],

            nn.Conv1d(channels, output_channels, 1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # [B, n_mels, T] → [B, 128, T]
        return self.features(x)