import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(self, input_features: int, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 4):
        super(TransformerNet, self).__init__()

        self.pe = PositionalEncoding(d_model=input_features)
        layer = nn.TransformerEncoderLayer(
            d_model = input_features,
            nhead = num_heads,
            dim_feedforward = hidden_size,
            batch_first = True
        )
        self.features = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # B x F x T
        x = x.permute(0, 2, 1)
        x = self.pe(x)
        x = self.features(x)     # B x T x F
        x = x.permute(0, 2, 1)   # B x F x T
        return x
