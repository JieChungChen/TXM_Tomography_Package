import torch
import torch.nn.functional as F
from torch import nn


class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_len=181):
        super().__init__()
        self.rel_pos = nn.Parameter(torch.randn(2 * max_len - 1, head_dim))  # [2L-1, head_dim]
        self.proj = nn.Linear(head_dim, 1, bias=False)  # 將每個相對位置的embedding投影成scalar bias
        self.max_len = max_len

    def forward(self, qlen, klen):
        range_q = torch.arange(qlen, device=self.rel_pos.device)[:, None]
        range_k = torch.arange(klen, device=self.rel_pos.device)[None, :]
        relative_indices = range_k - range_q + self.max_len - 1  # shift index to >= 0
        rel_embeddings = self.rel_pos[relative_indices]  # [qlen, klen, head_dim]
        bias = self.proj(rel_embeddings).squeeze(-1)  # [qlen, klen]
        return bias  # scalar bias per relative position


class MultiHeadAttentionWithRPE(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, max_len=500):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0

        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rpe = RelativePositionalEncoding(self.head_dim, max_len)

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)  # [B, n_head, T, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, n_head, T, T]

        # Add relative position bias
        rel_pos_bias = self.rpe(T, T)  # [T, T]
        attn_scores = attn_scores + rel_pos_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T] → broadcast

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [B, n_head, T, head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)  # [B, T, d_model]
        return self.out_proj(attn_output)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, rpe=True, max_len=181):
        super().__init__()
        self.rpe = rpe
        if rpe:
            self.attention = MultiHeadAttentionWithRPE(d_model, n_head, drop_prob, max_len)
        else:
            self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        if self.rpe:
            x = self.attention(x, mask)
        else:
            x = self.attention(x, x, x, key_padding_mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class Alignment_Block(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, num_heads, depth, drop_prob, num_projs, rpe=True):
        super().__init__()
        self.rpe = rpe
        self.pos_embed = nn.Parameter(torch.zeros(1, num_projs, embed_dim))
        self.att_blocks = nn.ModuleList([EncoderLayer(embed_dim, mlp_ratio*embed_dim, num_heads, drop_prob) 
                                         for _ in range(depth)])
        self.spt = SpatialTransformer([num_projs, embed_dim])
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x_0, max_shift=50, mask=None):
        B, N, L = x_0.shape

        x = x_0 + self.pos_embed
        for layer in self.att_blocks:
            x = layer(x, mask=mask)
        shift = self.mlp(x)

        flow = torch.zeros((B, 2, N, L), device=x_0.device)
        flow[:, 1] = (shift * max_shift / (L // 2)).expand(-1, -1, L)
        x = self.spt(x_0.unsqueeze(1), flow).reshape(B, N, L)
        return shift, x


class Alignment_Net(nn.Module):
    def __init__(self, embed_dim=512, num_att_blocks=4, num_align_blocks=2, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_projections = 181
        self.align_blocks = nn.ModuleList([Alignment_Block(embed_dim=embed_dim, 
                                                           mlp_ratio=mlp_ratio, 
                                                           num_heads=num_heads, 
                                                           depth=num_att_blocks, 
                                                           drop_prob=dropout,
                                                           num_projs = self.num_projections) 
                                           for _ in range(num_align_blocks)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        B, N, L = x.shape

        for i, layer in enumerate(self.align_blocks):
            shift, x = layer(x, mask=mask)
            if i == 0:
                total_shift = shift
            else:
                total_shift = total_shift + shift
        return total_shift, x
      

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # y, x
        grid = grid.type(torch.FloatTensor)
        for i in range(len(size)):
            grid[i] = 2 * (grid[i]/(size[i]-1)-0.5)
        grid = torch.unsqueeze(grid, 0)  # add batch
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False, padding_mode='border')
    

def sino_align_transformer_builder(configs=None):
    if configs is not None:
        model = Alignment_Net(
            embed_dim=configs.get('embed_dim', 512),
            num_att_blocks=configs.get('num_att_blocks', 4),
            num_align_blocks=configs.get('num_align_blocks', 2),
            num_heads=configs.get('num_heads', 4),
            mlp_ratio=configs.get('mlp_ratio', 4),
            dropout=configs.get('dropout', 0.1),
        )
    else:
        model = Alignment_Net()
    return model
    

if __name__ == "__main__":
    from torchinfo import summary
    model = Alignment_Net()
    summary(model, input_size=(2, 181, 512))