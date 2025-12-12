import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SinogramMAE(nn.Module):
    def __init__(self, img_size=512, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.):
        super().__init__()
        self.num_projections = 181
        self.img_size = img_size
        # 每個 projection (512) -> token 向量
        self.proj_embed = nn.Linear(img_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_projections + 1, embed_dim))

        # Encoder
        self.encoder = TransformerEncoder(depth=depth, embed_dim=embed_dim,
                                          num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_projections + 1, decoder_embed_dim))
        self.decoder = TransformerDecoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                                          num_heads=decoder_num_heads, mlp_ratio=mlp_ratio)

        # Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, img_size)  # 預測完整 projection

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 181, 512)
        B = x.shape[0]
        x = self.proj_embed(x) + self.pos_embed[:, 1:, :]   # 投影嵌入 + 位置
        cls_token = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder(x)
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x[:, 1:, :])  # (B, 181, 512)
        return x

# Transformer blocks (簡化版本)
class TransformerEncoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    model = SinogramMAE()
    summary(model, input_size=(1, 181, 512))