import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SinogramMAE(nn.Module):
    def __init__(self, img_size=512, embed_dim=768, depth=12, num_heads=12, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.):
        super().__init__()
        self.num_projections = 181
        self.img_size = img_size
        self.proj_embed = nn.Linear(img_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_projections, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_projections, decoder_embed_dim))

        # Encoder
        self.encoder = TransformerEncoder(depth=depth, embed_dim=embed_dim,
                                          num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          seq_len=self.num_projections)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder = HybridDecoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                                     num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                     seq_len=self.num_projections)

        # Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, img_size)  
        self.refine_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, max_mask_len=30, mask_len=None):
        # x: (B, 181, 512)
        if mask_len is not None:
            seq_len = 181 - mask_len
        else:
            seq_len = 181 - random.randint(0, max_mask_len)
        B = x.shape[0]
        x = self.proj_embed(x) + self.pos_embed
        x = self.encoder(x[:, :seq_len, :])

        x = self.decoder_embed(x)
        x = torch.cat([x, self.mask_token.repeat(B, 181 - seq_len, 1)], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x)          # (B, 181, 512)
        x = x.unsqueeze(1)                # (B, 1, 181, 512)
        x = self.refine_head(x)           # conv refine
        x = x.squeeze(1)                  # (B, 181, 512)
        return x
# ---------------- Relative Pos Encoding in Attention ----------------

class RelPosMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * seq_len - 1), num_heads)
        )  # (2L-1, H)
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # (L, L)
        relative_coords += seq_len - 1
        self.register_buffer("relative_position_index", relative_coords, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):  # x: (B, L, C)
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, L, H, D)
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        rel_pos_index = self.relative_position_index[:L, :L].reshape(-1)
        rel_pos_bias = self.relative_position_bias_table[rel_pos_index]
        rel_pos_bias = rel_pos_bias.view(L, L, self.num_heads).permute(2, 0, 1)  # (H, L, L)
        attn = attn + rel_pos_bias.unsqueeze(0)  # broadcast to (B, H, L, L)

        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, seq_len):
        super().__init__()
        self.attn = RelPosMultiheadAttention(embed_dim, num_heads, seq_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio, seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, seq_len) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio, seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, seq_len) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

class DepthwiseConvPyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.branch2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.branch3 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels)
        self.proj = nn.Conv1d(channels * 3, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.proj(out)
        return self.act(self.bn(out))


class HybridDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, seq_len):
        super().__init__()
        self.global_block = TransformerBlock(embed_dim, num_heads, mlp_ratio, seq_len)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.conv_pyramid = DepthwiseConvPyramid(embed_dim)
        self.local_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.global_block(x)
        local = self.local_norm(x).transpose(1, 2)
        local = self.conv_pyramid(local).transpose(1, 2)
        local = self.local_proj(local)
        return x + self.gate * local


class HybridDecoder(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio, seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridDecoderBlock(embed_dim, num_heads, mlp_ratio, seq_len)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def sinogram_mae_build(configs=None):
    if configs is not None:
        model = SinogramMAE(
                configs['img_size'],
                configs['embed_dim'],
                configs['depth'],
                configs['num_heads'],
                configs['decoder_embed_dim'],
                configs['decoder_depth'],
                configs['decoder_num_heads'],
                configs['mlp_ratio']
                            )
    else:
        model = SinogramMAE()
    return model


def grad_loss(pred, target):
    # Sobel kernels
    gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=pred.dtype, device=pred.device).view(1,1,3,3)
    gy = gx.transpose(-1, -2)
    def sobel(x):
        sx = F.conv2d(x, gx, padding=1)
        sy = F.conv2d(x, gy, padding=1)
        return torch.sqrt(sx*sx + sy*sy + 1e-6)
    return F.l1_loss(sobel(pred), sobel(target))


def helgason_ludwig_loss(sinogram, max_order=2):
    """
    sinogram: (B, P, T)  P=投影角數, T=detector長度
    angles: (P,)  以弧度表示
    """
    B, P, T = sinogram.shape
    device = sinogram.device
    angles = torch.linspace(0, torch.pi, P, device=device)
    t = torch.linspace(-1.0, 1.0, T, device=device)
    weights = torch.ones_like(t)
    sinogram = sinogram * weights.view(1, 1, -1)

    loss = 0.0
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)

    for n in range(max_order + 1):
        # moment: ∫ s^n g(θ, s) ds
        moment = torch.einsum('pt,bpt->bp', t.pow(n).unsqueeze(0).repeat(P, 1), sinogram)
        # build HL basis
        basis = []
        for m in range(n + 1):
            basis.append((cos_t ** (n - m)) * (sin_t ** m))
        Phi = torch.stack(basis, dim=1)  # (P, n+1)
        # least squares projection
        pinv = torch.linalg.pinv(Phi)
        coeff = torch.matmul(pinv, moment.transpose(0,1))          # (n+1, B)
        recon = torch.matmul(Phi, coeff).transpose(0,1)            # (B, P)
        loss += F.mse_loss(recon, moment)
    return loss / (max_order + 1)


if __name__ == "__main__":
    model = SinogramMAE()
    summary(model, input_size=(2, 181, 512))