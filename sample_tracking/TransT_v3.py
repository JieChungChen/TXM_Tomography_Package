import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from resnet import resnet50
import math, copy
from typing import Optional



class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # 預先計算常見尺寸的 position encoding
        common_sizes = [(16, 16), (32, 32)]

        for h, w in common_sizes:
            pos_enc = self._compute_pos_encoding_static(h, w)
            self.register_buffer(f'pos_cache_{h}x{w}', pos_enc)

        print(f"[PositionEmbeddingSine] Pre-computed position encodings for sizes: {common_sizes}")

    def _compute_pos_encoding_static(self, h, w):
        """
        Args:
            h, w: 特徵圖的高度和寬度

        Returns:
            pos: [1, C, H, W] 格式的 position encoding，C = 2 * num_pos_feats
        """
        # 創建座標網格（單一樣本）
        mask = torch.zeros((1, h, w), dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # 正規化座標
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 計算頻率項
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 計算 sin/cos position encoding
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 輸入特徵圖

        Returns:
            pos: [B, C, H, W] position encoding
        """
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # 嘗試從快取讀取
        cache_key = f'pos_cache_{h}x{w}'
        if hasattr(self, cache_key):
            cached_pos = getattr(self, cache_key)
            return cached_pos.expand(b, -1, -1, -1)

        # 快取未命中：執行動態計算（fallback）
        print(f"[Warning] Position encoding cache miss for size {h}x{w}, computing dynamically...")
        return self._compute_pos_encoding_dynamic(x)

    def _compute_pos_encoding_dynamic(self, x):
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class FeatureFusionNetwork(nn.Module):    
    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Encoder: 多層FeatureFusionLayer
        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder_layers = self._get_clones(featurefusion_layer, num_featurefusion_layers)
        
        # Decoder: 單層DecoderCFA
        self.decoder = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_featurefusion_layers

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search, pos_temp=None, pos_search=None):
        """
        Args:
            src_temp: [HW_t, B, C] template features (already flattened and permuted)
            src_search: [HW_s, B, C] search features (already flattened and permuted)
            pos_temp: [HW_t, B, C] position encoding for template
            pos_search: [HW_s, B, C] position encoding for search
        Returns:
            output: [B, C, HW_s] final search features
        """
        # Encoder: 多層特徵融合
        memory_temp = src_temp
        memory_search = src_search
        
        for layer in self.encoder_layers:
            memory_temp, memory_search = layer(
                memory_temp, memory_search,
                pos_temp=pos_temp, pos_search=pos_search
            )
        
        # Decoder: 最終的cross-attention refinement
        output = self.decoder(
            tgt=memory_search,
            memory=memory_temp,
            pos_temp=pos_temp,
            pos_search=pos_search
        )
        
        # Apply normalization
        output = self.decoder_norm(output)
        
        # [HW_s, B, C] -> [B, C, HW_s]
        output = output.permute(1, 2, 0)
        
        return output


class FeatureFusionLayer(nn.Module):    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # ECA: Ego-Context Augment (Self-Attention) for both branches
        self.self_attn_temp = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_search = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # CFA: Cross-Feature Augment (雙向Cross-Attention)
        self.cross_attn_t2s = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_s2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN for template branch
        self.linear1_t = nn.Linear(d_model, dim_feedforward)
        self.dropout_t = nn.Dropout(dropout)
        self.linear2_t = nn.Linear(dim_feedforward, d_model)
        
        # FFN for search branch
        self.linear1_s = nn.Linear(d_model, dim_feedforward)
        self.dropout_s = nn.Dropout(dropout)
        self.linear2_s = nn.Linear(dim_feedforward, d_model)
        
        # Layer Norms for template branch
        self.norm1_t = nn.LayerNorm(d_model)  # after self-attention
        self.norm2_t = nn.LayerNorm(d_model)  # after cross-attention
        self.norm3_t = nn.LayerNorm(d_model)  # after FFN
        
        # Layer Norms for search branch
        self.norm1_s = nn.LayerNorm(d_model)  # after self-attention
        self.norm2_s = nn.LayerNorm(d_model)  # after cross-attention
        self.norm3_s = nn.LayerNorm(d_model)  # after FFN
        
        # Dropouts
        self.dropout1_t = nn.Dropout(dropout)
        self.dropout2_t = nn.Dropout(dropout)
        self.dropout3_t = nn.Dropout(dropout)
        
        self.dropout1_s = nn.Dropout(dropout)
        self.dropout2_s = nn.Dropout(dropout)
        self.dropout3_s = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src_temp, src_search, pos_temp=None, pos_search=None):
        """
        執行順序：
        1. ECA: 兩個branch各自做self-attention
        2. CFA: 雙向cross-attention
        3. FFN: Feed-forward networks
        """
        
        # ========== ECA: Self-Attention ==========
        # Template self-attention
        q_t = k_t = self.with_pos_embed(src_temp, pos_temp)
        src_temp2 = self.self_attn_temp(q_t, k_t, value=src_temp)[0]
        src_temp = src_temp + self.dropout1_t(src_temp2)
        src_temp = self.norm1_t(src_temp)
        
        # Search self-attention
        q_s = k_s = self.with_pos_embed(src_search, pos_search)
        src_search2 = self.self_attn_search(q_s, k_s, value=src_search)[0]
        src_search = src_search + self.dropout1_s(src_search2)
        src_search = self.norm1_s(src_search)
        
        # ========== CFA: Cross-Attention (雙向) ==========
        # Template attending to Search
        src_temp2 = self.cross_attn_s2t(
            query=self.with_pos_embed(src_temp, pos_temp),
            key=self.with_pos_embed(src_search, pos_search),
            value=src_search
        )[0]
        
        # Search attending to Template
        src_search2 = self.cross_attn_t2s(
            query=self.with_pos_embed(src_search, pos_search),
            key=self.with_pos_embed(src_temp, pos_temp),
            value=src_temp
        )[0]
        
        # Update with cross-attention results
        src_temp = src_temp + self.dropout2_t(src_temp2)
        src_temp = self.norm2_t(src_temp)
        
        src_search = src_search + self.dropout2_s(src_search2)
        src_search = self.norm2_s(src_search)
        
        # ========== FFN: Feed-Forward Networks ==========
        # Template FFN
        src_temp2 = self.linear2_t(self.dropout_t(
            self.activation(self.linear1_t(src_temp))
        ))
        src_temp = src_temp + self.dropout3_t(src_temp2)
        src_temp = self.norm3_t(src_temp)
        
        # Search FFN
        src_search2 = self.linear2_s(self.dropout_s(
            self.activation(self.linear1_s(src_search))
        ))
        src_search = src_search + self.dropout3_s(src_search2)
        src_search = self.norm3_s(src_search)
        
        return src_temp, src_search


class DecoderCFALayer(nn.Module):    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Cross-attention: search queries template
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, pos_temp=None, pos_search=None):
        """
        Args:
            tgt: search features [HW_s, B, C]
            memory: template features [HW_t, B, C]
        """
        
        # Cross-attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos_search),
            key=self.with_pos_embed(memory, pos_temp),
            value=memory
        )[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(
            self.activation(self.linear1(tgt))
        ))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        return tgt


class TransT(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, frozen_backbone_layers=None):
        super().__init__()

        # if frozen_backbone_layers is None:
        #     frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']

        # Backbone（帶凍結層）
        self.backbone = resnet50(pretrained=False, output_layers=None, frozen_layers=frozen_backbone_layers)

        # 輸出凍結層資訊
        if frozen_backbone_layers:
            print(f"[TransT] Frozen backbone layers: {frozen_backbone_layers}")
        
        # 降維
        self.input_proj = nn.Conv2d(1024, d_model, kernel_size=1)
        
        # Position encoding
        self.position_embedding = PositionEmbeddingSine(d_model//2, normalize=True)
        
        # Feature Fusion layers
        self.featurefusion_network = FeatureFusionNetwork(
            d_model=d_model,
            nhead=nhead,
            num_featurefusion_layers=num_featurefusion_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(d_model, 2)  # object/no-object
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # 3-layer MLP
        
        # 初始化
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, template, search):
        """
        Args:
            template: [B, 1, H1, W1]
            search: [B, 1, H2, W2]
        Returns:
            out: dict with 'pred_logits' and 'pred_boxes'
        """
        # Backbone features
        feat_temp = self.backbone(template)
        feat_search = self.backbone(search)

        # Project to d_model dimensions
        src_temp = self.input_proj(feat_temp)
        src_search = self.input_proj(feat_search)
        
        # Position encodings
        pos_temp = self.position_embedding(src_temp)
        pos_search = self.position_embedding(src_search)
        
        # Flatten for transformer
        bs, c, h_t, w_t = src_temp.shape
        bs, c, h_s, w_s = src_search.shape
        
        src_temp = src_temp.flatten(2).permute(2, 0, 1)  # [HW_t, B, C]
        src_search = src_search.flatten(2).permute(2, 0, 1)  # [HW_s, B, C]
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        
        # Feature fusion
        fused_search = self.featurefusion_network(
            src_temp, src_search, pos_temp, pos_search
        )
        
        # Reshape back
        fused_search = fused_search.reshape(bs, c, h_s, w_s)
        
        # Predictions at each spatial location
        src_search_flat = fused_search.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Class and bbox predictions
        outputs_class = self.class_embed(src_search_flat)  # [B, HW, 2]
        outputs_coord = self.bbox_embed(src_search_flat).sigmoid()  # [B, HW, 4]
        
        # Reshape to spatial format
        outputs_class = outputs_class.permute(0, 2, 1).reshape(bs, 2, h_s, w_s)
        outputs_coord = outputs_coord.permute(0, 2, 1).reshape(bs, 4, h_s, w_s)
        
        return outputs_class, outputs_coord


class MLP(nn.Module):
    """簡單的 MLP - 用於 bbox regression head"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) 
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":
    # 測試
    model = TransT()
    template = torch.randn(2, 1, 128, 128)
    search = torch.randn(2, 1, 256, 256)
    
    cls_logits, bbox = model(template, search)
    print(f"Classification output shape: {cls_logits.shape}")
    print(f"Bbox output shape: {bbox.shape}")
    print(f"Bbox range: [{bbox.min():.3f}, {bbox.max():.3f}]")