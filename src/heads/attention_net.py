import torch.nn as nn


class AttentionCompleteNet(nn.Module):
    def __init__(
        self, output_features: int, hidden_size: int = 128
    ):
        # B x T x 128
        super(AttentionCompleteNet, self).__init__()
        self.query_projection = nn.Linear(
            in_features=1 * hidden_size, out_features=1 * hidden_size
        )
        self.to_answer = nn.Sequential(
            nn.Linear(
                in_features=hidden_size, out_features=hidden_size // 2
            ),  # to dense 64
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size // 2, out_features=hidden_size // 4
            ),  # to dense 32
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size // 4, out_features=output_features
            ),  # to dense nclasses
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # B x T x 128
        x = x.permute(0, 2, 1)  # B x 128 x T
        T = x.shape[2]
        q = x[:, :, T//2]  # B x 1? x 128
        q = self.query_projection(q)  # B x 1? x 128
        e = q.unsqueeze(1) @ x  # B x 1 x T
        a = self.softmax(e)  # B x 1 x T
        v = (a @ x.transpose(1, 2)).squeeze(1)  # B x 128
        return self.to_answer(v)
