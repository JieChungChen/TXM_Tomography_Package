import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(50, channels // 2))
        self.col_embed = nn.Parameter(torch.randn(50, channels // 2))
        
    def forward(self, tensor):
        h, w = tensor.shape[-2:]
        pos_h = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)
        pos_w = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)
        pos = torch.cat([pos_h, pos_w], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        return pos.repeat(tensor.shape[0], 1, 1, 1)
    
    
class ECA(nn.Module):
    """Efficient Channel Attention (channel-wise reweighting)"""
    def __init__(self, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,1,C)->(B,C,1)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B,C,1,1)
        return x * y.expand_as(x)


class CFA(nn.Module):
    """Cross Feature Attention (template <-> search interaction)"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, z):
        # x: search sequence, z: template sequence
        x = self.norm(x)
        z = self.norm(z)
        out, _ = self.cross_attn(x, z, z)
        return x + out


class TransformerBlock(nn.Module):
    """包含 Self-Attn, CFA, FFN"""
    def __init__(self, dim, heads=8, ffn_dim=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cfa = CFA(dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, z):
        # Self attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross attention
        x = self.cfa(x, z)
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x
    

class Backbone(nn.Module):
    def __init__(self, out_dim=256, use_eca=True):
        super().__init__()
        resnet = resnet50(pretrained=True)
        old_weight = resnet.conv1.weight.data.clone()   # (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.proj = nn.Conv2d(1024, out_dim, 1)
        self.eca = ECA() if use_eca else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)   # (B, 1024, H, W)
        x = self.eca(x)
        x = self.proj(x)
        return x


class TransT(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, ffn_dim=2048):
        super().__init__()
        self.backbone = Backbone(out_dim=dim, use_eca=True)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, ffn_dim) for _ in range(depth)])

        # 分類頭保持簡單
        self.cls_head = nn.Conv2d(dim, 2, kernel_size=1)

        # BBox頭：使用3層MLP（按照原始TransT）
        # 對feature map的每個位置應用MLP
        self.box_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, kernel_size=1)
        )

    def forward(self, template, search):
        # Feature extraction
        z_feat = self.backbone(template)  # (B,C,H,W)
        x_feat = self.backbone(search)

        # Flatten to sequence
        B, C, H, W = x_feat.shape
        z_seq = z_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        x_seq = x_feat.flatten(2).permute(0, 2, 1)

        # Transformer
        for blk in self.blocks:
            x_seq = blk(x_seq, z_seq)

        # Prediction
        f_map = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
        cls_logits = self.cls_head(f_map)

        # BBox prediction with sigmoid constraint (like original TransT)
        # 預測格式：(cx, cy, w, h) normalized to [0, 1]
        bbox = self.box_head(f_map).sigmoid()

        return cls_logits, bbox


if __name__ == "__main__":
    from torchinfo import summary
    model = TransT()
    summary(model, input_size=[(1, 1, 128, 128), (1, 1, 256, 256)])
