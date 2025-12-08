import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=181):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_len=500):
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
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, n_head, head_dim]
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
    # applied along the last dimension
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
    def __init__(self, ch, ffn_hidden, n_head, depth, drop_prob, rpe=True):
        super().__init__()
        self.rpe = rpe
        self.pos_emb = PositionalEncoding(ch)
        self.att_blocks = nn.ModuleList([EncoderLayer(ch, ffn_hidden, n_head, drop_prob) for _ in range(depth)])
        self.spt = SpatialTransformer([181, ch])
        self.mlp = nn.Sequential(
            nn.Linear(ch, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 

    def forward(self, x, mask=None):
        B, N, L = x.shape
        h = x
        if not self.rpe:
            h = self.pos_emb(x)
        for i, layer in enumerate(self.att_blocks):
            h = layer(h, mask=mask)
        shift = self.mlp(h)
        flow = torch.zeros((B, 2, N, L)).to(x.device)
        for proj in range(N):
            flow[:, 1, proj] = shift[:, proj]*50/256
        x = self.spt(x.unsqueeze(1), flow).reshape(B, N, L)
        return shift, x


class Alignment_Net(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ch = configs['ch']
        depth = configs['num_att_blocks']
        n_align = configs['num_align_blocks']
        dropout = configs['dropout']
        self.align_blocks = nn.ModuleList([Alignment_Block(ch, 2*ch, 4, depth, dropout) for _ in range(n_align)])

    def forward(self, x, mask=None):
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
    

class Filter_Back_Projection(nn.Module):
    def __init__(self, img_size, filtered=False):
        super().__init__()
        self.img_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_size))))
        self.filtered = filtered
        center = img_size//2
        x, y = torch.meshgrid(
            torch.arange(img_size) - center,
            torch.arange(img_size) - center,
            indexing='ij')
        self.register_buffer("hann", self.get_hann_filter(self.img_size_padded))
        self.register_buffer("x", x)
        self.register_buffer("y", y)
        recon_0 =self.empty_recon(img_size, torch.linspace(0, np.pi, 181))
        self.register_buffer("recon_0", recon_0)
        
    def empty_recon(self, img_size, angles):
        recon = torch.zeros((img_size, img_size))
        sino_empty = torch.ones((1, len(angles), img_size))
        if self.filtered:
            pad_width = (0, self.img_size_padded - img_size)
            sino_padded = torch.nn.functional.pad(sino_empty, pad_width, 'constant', 0)
            sino_fft = torch.fft.fft(sino_padded, dim=-1)
            sino_empty = torch.real(torch.fft.ifft(sino_fft * self.hann, dim=-1))[:, :, :img_size]
        for i, theta in enumerate(angles):
            t = self.x * torch.cos(theta+torch.pi/2) + self.y * torch.sin(theta+torch.pi/2)
            t_idx = torch.round(t + img_size//2).long()  
            valid = (t_idx >= 0) & (t_idx < img_size)
            recon[valid] += sino_empty[0, i, t_idx[valid]]
        return recon
    
    def forward(self, sino, angles, circle=True):
        B, n_proj, L = sino.shape
        center = L//2
        if self.filtered:
            pad_width = (0, self.img_size_padded - L)
            sino_padded = torch.nn.functional.pad(sino, pad_width, 'constant', 0)
            sino_fft = torch.fft.fft(sino_padded, dim=-1)
            sino = torch.real(torch.fft.ifft(sino_fft * self.hann, dim=-1))[:, :, :L]
        
        recon = torch.zeros((B, L, L), device=sino.device)
        for i, theta in enumerate(angles):
            # 計算各theta下reconstruction每個pixel投影在sinogram上的座標
            t = self.x * torch.cos(theta+torch.pi/2) + self.y * torch.sin(theta+torch.pi/2)
            t_norm = 2 * t / (L - 1)  # [-1, 1] for grid_sample

            # grid_sample 要求 shape: [B, 1, H, W]
            t_norm = t_norm.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 1, H, W]

            # 提取該角度對應的 sinogram
            sino_i = sino[:, i, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

            # 建立 grid: (x = 投影位置, y = dummy 0)
            grid = torch.stack((t_norm, torch.zeros_like(t_norm)), dim=-1)  # [B, 1, H, W, 2]

            # 使用 grid_sample 把 sinogram 投影值拉回影像座標
            sampled = F.grid_sample(sino_i.float(), grid.squeeze(), mode='bilinear', align_corners=True, padding_mode='zeros')
            recon += sampled[:, 0]

        if circle:
            Y, X = np.ogrid[:L, :L]
            dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
            mask = dist_from_center > center
            recon[:, mask] = 0
        else:
            recon /= self.recon_0
        return recon
    
    def get_hann_filter(self, img_size):
        n = torch.concatenate((torch.arange(1, img_size / 2 + 1, 2, dtype=int),
                               torch.arange(img_size / 2 - 1, 0, -2, dtype=int)))
        f = torch.zeros(img_size)
        f[0] = 0.25
        f[1::2] = -1 / (torch.pi * n) ** 2
        ramp = 2 * torch.real(torch.fft.fft(f))
        hann = torch.from_numpy(np.hanning(img_size))
        return ramp * torch.fft.fftshift(hann)


class FBPLoss(nn.Module):
    def __init__(self, img_size, filtered=True):
        super().__init__()
        self.fbp = Filter_Back_Projection(img_size, filtered)
        self.tv = TVLoss()
        self.mse = nn.MSELoss()

    def forward(self, sino_pred, sino_true, angles):
        recon_pred = self.fbp(sino_pred, angles)
        recon_true = self.fbp(sino_true, angles)
        # recon_true = gaussian_blur(recon_true, [15, 15])

        loss_mse = self.mse(recon_pred, recon_true)
        loss_tv = self.tv(recon_pred, sino=False)
        return loss_mse+loss_tv
    
    def get_fbp_results(self, sino_pred, sino_true, angles, circle):
        recon_pred = self.fbp(sino_pred, angles, circle)
        recon_true = self.fbp(sino_true, angles, circle)
        return recon_pred, recon_true
    

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sino=True):
        # assume x: [B, H, W]
        dh = torch.abs(x[:, :-1, :] - x[:, 1:, :])
        dw = torch.abs(x[:, :, :-1] - x[:, :, 1:])
        if sino:
            loss =  dh.mean()
        else:
            weight_h = torch.exp(-torch.abs(dh))
            weight_w = torch.exp(-torch.abs(dw))
            loss = (weight_h * dh**2).mean() + (weight_w * dw**2).mean()
        return loss
        

def shift_range_penalty(shift_pred, max_shift=2):
    # shift_pred: shape (B, N), predicted shift values
    over_upper = torch.relu(shift_pred - max_shift)
    under_lower = torch.relu(-max_shift - shift_pred)
    penalty = over_upper + under_lower
    return torch.mean(penalty**2)
    
