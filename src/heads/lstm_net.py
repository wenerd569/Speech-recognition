import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, output_features: int, hidden_size: int = 64, num_layers: int = 2):
        super(LSTMNet, self).__init__()
        self.output_features = 2 * hidden_size

        self.features = nn.LSTM(
            input_size = output_features,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, x):
        # B x C x F' x T'
        x = x.permute(0, 3, 1, 2).flatten(2)  # B x T' x (C x F')
        x, _ = self.features(x)
        x = x.permute(0, 2, 1).unsqueeze(2)
        return x

