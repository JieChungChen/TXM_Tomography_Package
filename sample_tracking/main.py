import argparse, os, yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from preprocess_transt import TransTTrackingDataset
from TransT_v3 import TransT
from loss_v1 import build_transt_loss, prepare_targets, box_cxcywh_to_xyxy, generalized_box_iou


def evaluate_on_val_set(model, val_dataset, device, criterion, epoch):
    model.eval()

    # 創建測試集 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # 測試時可以用較大 batch size
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    total_loss = 0
    loss_dict_sum = {'loss_cls': 0, 'loss_bbox': 0, 'loss_giou': 0, 'iou': 0}
    n_samples = 0

    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1} - 測試集驗證")
    print(f"{'='*70}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="  Validating", dynamic_ncols=True)):
            # 準備輸入
            template = batch['template_images'].to(device, non_blocking=True)
            search = batch['search_images'].to(device, non_blocking=True)

            # 模型推理
            cls_logits, pred_boxes = model(template, search)

            # 準備損失函數輸入
            outputs = {
                'pred_logits': cls_logits,
                'pred_boxes': pred_boxes
            }
            targets = prepare_targets(batch)

            # 計算損失
            loss, loss_dict = criterion(outputs, targets)

            batch_size = template.size(0)
            total_loss += loss.item() * batch_size
            for key in loss_dict:
                loss_dict_sum[key] += loss_dict[key].item() * batch_size
            n_samples += batch_size

    # 計算平均指標
    val_metrics = {
        'val_total_loss': total_loss / n_samples,
        'val_cls_loss': loss_dict_sum['loss_cls'] / n_samples,
        'val_bbox_loss': loss_dict_sum['loss_bbox'] / n_samples,
        'val_giou_loss': loss_dict_sum['loss_giou'] / n_samples,
        'val_iou': loss_dict_sum['iou'] / n_samples,
    }

    # 打印結果
    print(f"  測試集樣本數: {n_samples}")
    print(f"  Total Loss:   {val_metrics['val_total_loss']:.4f}")
    print(f"  Cls Loss:     {val_metrics['val_cls_loss']:.4f}")
    print(f"  BBox Loss:    {val_metrics['val_bbox_loss']:.4f}")
    print(f"  GIoU Loss:    {val_metrics['val_giou_loss']:.4f}")
    print(f"  IoU:          {val_metrics['val_iou']:.4f}")
    print(f"{'='*70}\n")

    model.train()

    # 清理驗證過程的 GPU cache
    del val_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return val_metrics


def get_args_parser():
    parser = argparse.ArgumentParser('TransT Tracking Training', add_help=False)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--configs', default='configs/transt_train.yml', type=str)
    return parser


def visualize_prediction(model, dataset, device, save_path, epoch):
    """在隨機訓練樣本上進行推理並可視化預測結果"""
    model.eval()

    # 隨機選擇一個樣本
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]

    # 準備輸入
    template = sample['template_images'].unsqueeze(0).to(device)
    search = sample['search_images'].unsqueeze(0).to(device)
    gt_bbox = sample['search_anno'].numpy()  # normalized cxcywh

    with torch.no_grad():
        # 模型推理
        cls_logits, pred_boxes = model(template, search)

        # 使用最大信心位置提取預測 (inference 方式)
        cls_prob = torch.softmax(cls_logits, dim=1)[0, 0]  # [H, W] object prob (class 0)
        max_pos = torch.unravel_index(cls_prob.argmax(), cls_prob.shape)
        y_max, x_max = max_pos[0].item(), max_pos[1].item()

        # 提取最大信心位置的預測
        pred_bbox = pred_boxes[0, :, y_max, x_max].cpu()
        max_conf = cls_prob[y_max, x_max].item()

        # 計算GIoU
        from loss_v1 import box_cxcywh_to_xyxy, generalized_box_iou
        gt_bbox_tensor = torch.from_numpy(gt_bbox).unsqueeze(0).float()
        pred_bbox_tensor = pred_bbox.unsqueeze(0)

        gt_xyxy = box_cxcywh_to_xyxy(gt_bbox_tensor)
        pred_xyxy = box_cxcywh_to_xyxy(pred_bbox_tensor)

        # 確保bbox有效
        pred_x1y1 = pred_xyxy[:, :2]
        pred_x2y2 = torch.max(pred_xyxy[:, 2:], pred_xyxy[:, :2] + 1e-6)
        pred_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=1)

        gt_x1y1 = gt_xyxy[:, :2]
        gt_x2y2 = torch.max(gt_xyxy[:, 2:], gt_xyxy[:, :2] + 1e-6)
        gt_xyxy = torch.cat([gt_x1y1, gt_x2y2], dim=1)

        giou, iou = generalized_box_iou(pred_xyxy, gt_xyxy)
        giou_value = giou[0, 0].item()
        giou_loss = 1 - giou_value

        pred_bbox = pred_bbox.numpy()

    # 準備可視化
    search_img = sample['search_images'].squeeze().numpy()
    template_img = sample['template_images'].squeeze().numpy()

    # 轉換 bbox 到像素座標 (search_size = 256)
    search_size = 256

    # Ground truth bbox
    gt_cx, gt_cy, gt_w, gt_h = gt_bbox * search_size
    gt_x1, gt_y1 = gt_cx - gt_w/2, gt_cy - gt_h/2

    # Predicted bbox
    pred_cx, pred_cy, pred_w, pred_h = pred_bbox * search_size
    pred_x1, pred_y1 = pred_cx - pred_w/2, pred_cy - pred_h/2

    # 創建可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Template
    ax1.imshow(template_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Template (Epoch {epoch+1})', fontsize=14)
    ax1.axis('off')

    # Search with predictions
    ax2.imshow(search_img, cmap='gray', vmin=0, vmax=1)

    # Ground truth (green)
    gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_w, gt_h,
                               linewidth=2, edgecolor='green', facecolor='none',
                               label='Ground Truth')
    ax2.add_patch(gt_rect)

    # Prediction (red)
    pred_rect = patches.Rectangle((pred_x1, pred_y1), pred_w, pred_h,
                                 linewidth=2, edgecolor='red', facecolor='none',
                                 linestyle='--', label='Prediction')
    ax2.add_patch(pred_rect)

    ax2.set_title(f'Search Region (Epoch {epoch+1})', fontsize=14)
    ax2.legend()
    ax2.axis('off')

    # 添加預測資訊
    pred_info = f'Sample: {sample["seq_name"]}, Frame: {sample["frame_id"]}\n'
    pred_info += f'GT: ({gt_cx:.1f}, {gt_cy:.1f}, {gt_w:.1f}, {gt_h:.1f})\n'
    pred_info += f'Pred: ({pred_cx:.1f}, {pred_cy:.1f}, {pred_w:.1f}, {pred_h:.1f})\n'
    pred_info += f'Confidence: {max_conf:.4f}\n'
    pred_info += f'GIoU: {giou_value:.4f}, GIoU Loss: {giou_loss:.4f}'
    plt.figtext(0.02, 0.02, pred_info, fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 清理可視化過程的 GPU tensors
    del template, search, cls_logits, pred_boxes, cls_prob
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model.train()
    print(f"Visualization saved to: {save_path}")


def main(args):
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)

    trn_conf = configs['training_settings']
    data_conf = configs['data_settings']
    device = torch.device(trn_conf['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(trn_conf['model_save_dir'], exist_ok=True)
    os.makedirs('temp', exist_ok=True)

    # 使用新的 TransT 實現
    model = TransT().to(device)
    if trn_conf.get('ckpt_path') and trn_conf['ckpt_path'] != '~':
        ckpt_full_path = os.path.join(trn_conf['model_save_dir'], trn_conf['ckpt_path'])
        if os.path.exists(ckpt_full_path):
            model.load_state_dict(torch.load(ckpt_full_path, map_location=device), strict=False)
            print(f"Loaded checkpoint: {ckpt_full_path}")

    # 使用新的數據集 - 訓練集
    dataset = TransTTrackingDataset(
        data_conf['trn_data_dir'],
        template_size=trn_conf.get('template_size', 128),
        search_size=trn_conf.get('search_size', 256),
        mode='train'
    )
    dataloader = DataLoader(dataset, batch_size=trn_conf['batch_size'], shuffle=True,
                            num_workers=trn_conf.get('num_workers', 4), drop_last=True,
                            pin_memory=True)

    # 載入測試集（如果有提供）
    val_dataset = None
    val_data_dir = data_conf.get('val_data_dir')
    if val_data_dir and os.path.exists(val_data_dir):
        print(f"Loading validation dataset from: {val_data_dir}")
        val_dataset = TransTTrackingDataset(
            val_data_dir,
            template_size=trn_conf.get('template_size', 128),
            search_size=trn_conf.get('search_size', 256),
            mode='test'  # 測試模式：不做 augmentation
        )
        print(f"Validation dataset size: {len(val_dataset)} samples")
    else:
        print("No validation dataset provided, skipping validation")

    # 使用新的損失函數（按照原作TransT配置）
    loss_conf = configs.get('loss_settings', {})
    criterion = build_transt_loss(
        cls_weight=loss_conf.get('cls_weight', 2.0),
        bbox_weight=loss_conf.get('bbox_weight', 5.0),
        giou_weight=loss_conf.get('giou_weight', 2.0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=trn_conf['base_lr'], weight_decay=trn_conf['weight_decay'])

    # Cosine Annealing 學習率調度器
    scheduler = None
    warmup_epochs = 0
    if trn_conf.get('use_cosine_annealing', False):
        warmup_epochs = trn_conf.get('warmup_epochs', 5)
        T_max = trn_conf.get('T_max') or trn_conf['n_epochs']
        eta_min = trn_conf.get('eta_min', trn_conf['base_lr'] * 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min
        )
        print(f"Cosine Annealing enabled: warmup={warmup_epochs} epochs, T_max={T_max}, eta_min={eta_min}")

    scaler = torch.amp.GradScaler()

    # 梯度累積設定
    accumulation_steps = trn_conf.get('accumulation_steps', 1)
    effective_batch_size = trn_conf['batch_size'] * accumulation_steps
    print(f"Gradient Accumulation: {accumulation_steps} steps")
    print(f"Actual batch_size: {trn_conf['batch_size']}, Effective batch_size: {effective_batch_size}")

    model.train()
    for epoch in range(trn_conf['n_epochs']):
        if scheduler is not None and epoch < warmup_epochs:
            warmup_lr = trn_conf['base_lr'] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        total_loss = 0
        loss_dict_sum = {'loss_cls': 0, 'loss_bbox': 0, 'loss_giou': 0, 'iou': 0}
        current_lr = optimizer.param_groups[0]['lr']

        with tqdm(dataloader, dynamic_ncols=True, desc=f'Epoch {epoch+1}/{trn_conf["n_epochs"]} [LR: {current_lr:.2e}]') as tqdmDataLoader:
            for batch_idx, batch in enumerate(tqdmDataLoader):
                # 只在累積步驟的開始清空梯度
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                    template = batch['template_images'].to(device, non_blocking=True)
                    search = batch['search_images'].to(device, non_blocking=True)

                    cls_logits, pred_bbox = model(template, search)

                    outputs = {
                        'pred_logits': cls_logits,
                        'pred_boxes': pred_bbox
                    }

                    targets = prepare_targets(batch)
                    loss, loss_dict = criterion(outputs, targets)

                    # 梯度累積：將 loss 除以累積步數
                    loss = loss / accumulation_steps

                total_loss += loss.item() * accumulation_steps  # 記錄原始 loss
                for key in loss_dict:
                    loss_dict_sum[key] += loss_dict[key].item()

                # Backward
                scaler.scale(loss).backward()

                # 只在累積步驟結束時更新參數
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(tqdmDataLoader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf.get('grad_clip', 1.0))
                    scaler.step(optimizer)
                    scaler.update()

                # 更新進度條
                avg_losses = {k: v / (batch_idx + 1) for k, v in loss_dict_sum.items()}
                tqdmDataLoader.set_postfix({
                    "loss": total_loss / (batch_idx + 1),
                    "cls": avg_losses['loss_cls'],
                    "bbox": avg_losses['loss_bbox'],
                    "giou": avg_losses['loss_giou'],
                    "iou": avg_losses['iou']
                })

        # 保存模型
        if (epoch + 1) % trn_conf.get('save_n_epochs', 20) == 0:
            ckpt_path = os.path.join(trn_conf['model_save_dir'], f'transt_ep{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

            # 在測試集上進行驗證
            if val_dataset is not None:
                val_metrics = evaluate_on_val_set(model, val_dataset, device, criterion, epoch)

                val_log_path = os.path.join(trn_conf['model_save_dir'], 'validation_log.txt')
                with open(val_log_path, 'a') as f:
                    f.write(f"Epoch {epoch+1}:\n")
                    for key, value in val_metrics.items():
                        f.write(f"  {key}: {value:.4f}\n")
                    f.write("\n")
                print(f"Validation results saved to {val_log_path}")

                del val_metrics
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # 可視化訓練進度
        try:
            vis_save_path = f"temp/training_progress_epoch_{epoch+1:03d}.png"
            visualize_prediction(model, dataset, device, vis_save_path, epoch)
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Training continues...")
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 更新學習率（warmup 後才使用 scheduler）
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        # 打印訓練統計
        print(f"Epoch {epoch+1} - LR: {current_lr:.2e}, "
              f"Total: {total_loss/len(dataloader):.4f}, "
              f"Cls: {loss_dict_sum['loss_cls']/len(dataloader):.4f}, "
              f"BBox: {loss_dict_sum['loss_bbox']/len(dataloader):.4f}, "
              f"GIoU: {loss_dict_sum['loss_giou']/len(dataloader):.4f}, "
              f"IoU: {loss_dict_sum['iou']/len(dataloader):.4f}")

        # 每個 epoch 結束清理 GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)