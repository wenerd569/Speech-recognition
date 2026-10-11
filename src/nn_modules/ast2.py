"""
AST: Audio Spectrogram Transformer (arXiv:2104.01778)
Инициализация весами DeiT-base distilled 384 (ImageNet) через timm.

pip install torch torchaudio timm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm


class ASTNet(nn.Module):
    def __init__(
        self,
        n_mels,
        target_T,
        label_dim=527,
        fstride=10,
        tstride=10,
        imagenet_pretrain=True,
        model_name="deit_base_distilled_patch16_384",
    ):
        super().__init__()
        v = timm.create_model(model_name, pretrained=imagenet_pretrain)
        dim = v.embed_dim                                  # 768
        orig_h, orig_w = v.patch_embed.grid_size           # 24, 24

        # число патчей по частоте / времени при перекрытии
        f_dim = (n_mels - 16) // fstride + 1
        t_dim = (target_T - 16) // tstride + 1

        # --- 1) patch embedding: 3 канала -> 1 канал, новый stride ---
        proj = nn.Conv2d(1, dim, kernel_size=16, stride=(fstride, tstride))
        if imagenet_pretrain:
            old = v.patch_embed.proj
            proj.weight = nn.Parameter(old.weight.detach().sum(dim=1, keepdim=True))
            proj.bias = nn.Parameter(old.bias.detach().clone())
        self.proj = proj

        # --- 2) positional embedding: обрезка по частоте, интерполяция по времени ---
        pos = v.pos_embed.detach()                         # (1, 2+576, 768)
        head_tokens, grid = pos[:, :2], pos[:, 2:]
        grid = grid.reshape(1, orig_h, orig_w, dim).permute(0, 3, 1, 2)  # (1,768,24,24)
        if imagenet_pretrain:
            # время
            if t_dim <= orig_w:
                s = orig_w // 2 - t_dim // 2
                grid = grid[:, :, :, s:s + t_dim]
            else:
                grid = F.interpolate(grid, size=(orig_h, t_dim), mode="bilinear", align_corners=False)
            # частота
            if f_dim <= orig_h:
                s = orig_h // 2 - f_dim // 2
                grid = grid[:, :, s:s + f_dim, :]
            else:
                grid = F.interpolate(grid, size=(f_dim, t_dim), mode="bilinear", align_corners=False)
            grid = grid.reshape(1, dim, f_dim * t_dim).transpose(1, 2)
            self.pos_embed = nn.Parameter(torch.cat([head_tokens, grid], dim=1))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 + f_dim * t_dim, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 3) остальное берём из DeiT как есть ---
        self.cls_token = v.cls_token
        self.dist_token = v.dist_token
        self.blocks = v.blocks
        self.norm = v.norm

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, label_dim))

    def forward(self, x):
        """x: (B, time, freq) лог-мел fbank, нормированный."""
        x = x.unsqueeze(1).transpose(2, 3)                 # (B,1,freq,time)
        x = self.proj(x).flatten(2).transpose(1, 2)        # (B, N, 768)
        B = x.shape[0]
        x = torch.cat(
            [self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x], dim=1
        )
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)                            # логиты
