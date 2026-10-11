import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_f: int,
        input_t: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        patch_size: int = 16,
        stride: int = 10,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        # Размер сетки патчей для ожидаемого размера входа
        grid_f = (input_f - patch_size) // stride + 1
        grid_t = (input_t - patch_size) // stride + 1

        # Обучаемые двумерные позиционные эмбеддинги
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, grid_f, grid_t)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Специальный токен для классификации
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        # B × D × grid_f × grid_t

        if x.shape[2:] != self.pos_embed.shape[2:]:
            pos = F.interpolate(
                self.pos_embed,
                size=x.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
        else:
            pos = self.pos_embed

        x = x + pos

        x = x.flatten(2).transpose(1, 2)
        # B × N × D

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        # B × (N + 1) × D

        x = self.encoder(x)

        # Берём выход специального CLS-токена
        x = x[:, 0]

        return self.classifier(x)