import torch.nn as nn

class GruNet(nn.Module):
    def __init__(self, input_features: int, hidden_size: int = 64, num_layers: int = 2):
        super(GruNet, self).__init__()
        self.output_features = 2 * hidden_size

        self.features = nn.GRU(
            input_size = input_features,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, x):
        # B x F x T
        x = x.permute(0, 2, 1)   # B x T x F
        x, _ = self.features(x)  # B x T x 128
        return x