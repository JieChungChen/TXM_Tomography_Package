"""
TransT Loss Implementation - 基於原作者代碼，適配空間輸出格式
完全按照原作者設計，只調整輸入格式處理

模型輸出格式:
- pred_logits: [B, 2, H, W] - 空間分類輸出
- pred_boxes: [B, 4, H, W] - 空間回歸輸出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def box_cxcywh_to_xyxy(bbox):
    """轉換 bbox 從 (cx,cy,w,h) 到 (x1,y1,x2,y2)"""
    cx, cy, w, h = bbox.unbind(-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """計算兩組 bbox 的 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    計算 GIoU
    
    Returns:
        giou: [N, M] GIoU 矩陣
        iou: [N, M] IoU 矩陣
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / (area + 1e-6)
    
    return giou, iou


class TrackingMatcher(nn.Module):
    """原作者的 Matcher: 選擇 bbox 內所有點作為正樣本"""
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict 包含 'pred_logits' [B, C, H, W] 和 'pred_boxes' [B, 4, H, W]
            targets: list of dicts, 每個包含 'boxes' [1, 4] (normalized cxcywh)
        
        Returns:
            list of tuples (index_i, index_j)
        """
        indices = []
        
        # 從空間格式獲取尺寸
        bs = outputs["pred_logits"].shape[0]
        H, W = outputs["pred_logits"].shape[2:]
        num_queries = H * W
        
        for i in range(bs):
            # 獲取目標框座標 (normalized)
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item()
            cy = cy.item()
            w = w.item()
            h = h.item()
            
            # 轉換為 xyxy
            xmin = cx - w/2
            ymin = cy - h/2
            xmax = cx + w/2
            ymax = cy + h/2
            
            # 映射到特徵圖
            Xmin = int(np.ceil(xmin * W))
            Ymin = int(np.ceil(ymin * H))
            Xmax = int(np.ceil(xmax * W))
            Ymax = int(np.ceil(ymax * H))
            
            # 邊界檢查
            Xmin = max(0, min(Xmin, W - 1))
            Ymin = max(0, min(Ymin, H - 1))
            Xmax = max(0, min(Xmax, W))
            Ymax = max(0, min(Ymax, H))
            
            # 確保至少1像素
            if Xmin == Xmax:
                Xmax = min(Xmax + 1, W)
            if Ymin == Ymax:
                Ymax = min(Ymax + 1, H)
            
            # 選擇 bbox 內的點
            a = np.arange(0, num_queries, 1)
            b = a.reshape([H, W])
            c = b[Ymin:Ymax, Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            
            indices.append((c, d))
        
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


class SetCriterion(nn.Module):
    """TransT Loss - 完全按照原作者實現"""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 類別權重: [前景, 背景]
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """分類損失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, C, H, W]
        
        # Flatten: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = src_logits.shape
        src_logits = src_logits.flatten(2).permute(0, 2, 1)
        
        # 獲取匹配索引
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        
        # 創建目標: 默認背景類
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        target_classes[idx] = target_classes_o
        
        # Cross-Entropy Loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), 
                                  target_classes, 
                                  self.empty_weight)
        
        losses = {'loss_cls': loss_ce}
        
        if log:
            losses['class_error'] = 100 - self._accuracy(src_logits[idx], target_classes_o)[0]
        
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """回歸損失"""
        assert 'pred_boxes' in outputs
        
        # Flatten: [B, 4, H, W] -> [B, H*W, 4]
        pred_boxes = outputs['pred_boxes']
        B, _, H, W = pred_boxes.shape
        pred_boxes = pred_boxes.flatten(2).permute(0, 2, 1)
        
        # 獲取匹配的預測
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(pred_boxes.device)
        
        # L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # GIoU Loss
        giou, iou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        
        return losses
    
    def _accuracy(self, output, target):
        """計算準確率"""
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target)
            acc = correct.sum() / target.size(0) * 100
        return [acc]
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)
    
    def forward(self, outputs, targets):
        """計算總損失"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Matcher
        indices = self.matcher(outputs_without_aux, targets)
        
        # 計算正樣本數
        num_boxes_pos = sum(len(t[0]) for t in indices)
        num_boxes_pos = torch.as_tensor([num_boxes_pos], 
                                       dtype=torch.float, 
                                       device=next(iter(outputs.values())).device)
        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()
        
        # 計算損失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))
        
        # 加權總損失 - 確保從 tensor 開始
        total_loss = (
            losses['loss_cls'] * self.weight_dict['loss_cls'] +
            losses['loss_bbox'] * self.weight_dict['loss_bbox'] +
            losses['loss_giou'] * self.weight_dict['loss_giou']
        )
        
        return total_loss, losses


def build_matcher():
    """構建 matcher"""
    return TrackingMatcher()


def build_transt_loss(cls_weight=8.334, bbox_weight=5.0, giou_weight=2.0, eos_coef=0.0625):
    """構建 TransT 損失函數（原作者參數）"""
    num_classes = 1
    matcher = build_matcher()
    
    weight_dict = {
        'loss_cls': cls_weight,
        'loss_bbox': bbox_weight,
        'loss_giou': giou_weight
    }
    
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses
    ).to('cuda:0')
    
    return criterion


def prepare_targets(batch_data):
    """
    將 DataLoader 輸出轉換為 loss 需要的格式
    
    Args:
        batch_data: dict 包含 'search_anno' [B, 4]
    
    Returns:
        targets: list of dicts
    """
    batch_size = batch_data['search_anno'].shape[0]
    targets = []
    
    for i in range(batch_size):
        target = {
            'labels': torch.tensor([0], dtype=torch.int64, 
                                  device=batch_data['search_anno'].device),
            'boxes': batch_data['search_anno'][i:i+1]
        }
        targets.append(target)
    
    return targets